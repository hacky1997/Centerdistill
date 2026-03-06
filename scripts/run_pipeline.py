"""
scripts/run_pipeline.py — End-to-end CenterDistill pipeline.

Usage (Google Colab / GPU instance)
------------------------------------
    python scripts/run_pipeline.py \
        --mlqa_root MLQA_V1 \
        --output_dir outputs/centerdistill_seed42 \
        --seed 42

All hyperparameters (K, lambda, temperature, thresholds) are derived
from the training data — no manual tuning is required.
"""

import os
import sys
import json
import argparse
import numpy as np
import torch

# Make the package importable from scripts/
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from centerdistill.config   import BASE_CFG, lock_seed, derive_hyperparameters, save_config
from centerdistill.data     import load_en_en, load_en_es, load_en_de, evaluate_qa
from centerdistill.cluster  import encode_questions, induce_centers, \
                                   compute_teacher_distributions, cluster_quality_report
from centerdistill.model    import CenterDistillModel, CenterDistillTrainer, \
                                   build_training_args, patch_to_hf_qa_model
from centerdistill.evaluate import evaluate_behaviour, error_analysis, bootstrap_ci
from centerdistill.visualize import (
    plot_tsne, plot_cluster_summary, plot_system_performance,
    plot_metrics_heatmap, plot_silhouette_sweep, plot_k_ablation,
)


def parse_args():
    p = argparse.ArgumentParser(description="CenterDistill end-to-end pipeline")
    p.add_argument("--mlqa_root",  default="MLQA_V1",
                   help="Path to extracted MLQA_V1 directory")
    p.add_argument("--output_dir", default="outputs/centerdistill_seed42",
                   help="Directory for all saved artefacts")
    p.add_argument("--seed",       type=int, default=42)
    p.add_argument("--skip_train", action="store_true",
                   help="Skip training; load model from output_dir/centerdistill")
    p.add_argument("--skip_figs",  action="store_true",
                   help="Skip figure generation")
    return p.parse_args()


def main():
    args = parse_args()
    lock_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    CFG = dict(BASE_CFG)
    CFG["seed"]       = args.seed
    CFG["output_dir"] = args.output_dir

    bl_dir = os.path.join(args.output_dir, "baseline_en")
    cd_dir = os.path.join(args.output_dir, "centerdistill")
    os.makedirs(bl_dir, exist_ok=True)
    os.makedirs(cd_dir, exist_ok=True)

    # ── 1. Download MLQA if needed ───────────────────────────────────────────
    if not os.path.isdir(args.mlqa_root):
        print("Downloading MLQA …")
        os.system("curl -L -o MLQA_V1.zip "
                  "https://dl.fbaipublicfiles.com/MLQA/MLQA_V1.zip")
        os.system("unzip -q MLQA_V1.zip && rm MLQA_V1.zip")

    # ── 2. Load data ─────────────────────────────────────────────────────────
    print("\n[1/9] Loading data …")
    train_en, val_en, test_en = load_en_en(args.mlqa_root)
    dev_es,   test_es         = load_en_es(args.mlqa_root)
    dev_de,   test_de         = load_en_de(args.mlqa_root)
    print(f"  en-en  train={len(train_en)} val={len(val_en)} test={len(test_en)}")
    print(f"  en-es  dev={len(dev_es)} test={len(test_es)}")
    print(f"  en-de  dev={len(dev_de)} test={len(test_de)}")

    # ── 3. Cache baseline model ──────────────────────────────────────────────
    print("\n[2/9] Caching baseline model (SQuAD2 pre-trained) …")
    from transformers import AutoTokenizer, AutoModelForQuestionAnswering
    tokenizer = AutoTokenizer.from_pretrained(CFG["base_model"])
    if not os.path.exists(os.path.join(bl_dir, "config.json")):
        model_bl = AutoModelForQuestionAnswering.from_pretrained(CFG["base_model"])
        model_bl.save_pretrained(bl_dir)
        tokenizer.save_pretrained(bl_dir)
        del model_bl
        torch.cuda.empty_cache()
        print(f"  ✅ Cached → {bl_dir}")
    else:
        print(f"  ✅ Already cached at {bl_dir}")

    # ── 4. Compute embeddings and derive hyperparameters ─────────────────────
    print("\n[3/9] Computing LaBSE embeddings …")
    cluster_pool   = train_en[:CFG["n_cluster"]]
    question_texts = [ex["question"] for ex in cluster_pool]
    q_embs         = encode_questions(question_texts, seed=args.seed)

    from sklearn.preprocessing import normalize
    q_embs_norm = normalize(q_embs, norm="l2")

    print("\n[4/9] Deriving hyperparameters from data …")
    derived = derive_hyperparameters(q_embs_norm, seed=args.seed)
    CFG.update({k: v for k, v in derived.items()
                if k not in ("centroids", "silhouette_by_k", "gold_dist")})
    centroids = derived["centroids"]

    save_config(CFG, os.path.join(args.output_dir, "config.json"))
    np.save(os.path.join(args.output_dir, "centroids.npy"), centroids)
    print(f"  K={CFG['K']} | τ={CFG['temperature']} | "
          f"λ={CFG['lambda_kl']} | τ_conf={CFG['tau_conf']:.4f}")

    # ── 5. Cluster quality analysis ──────────────────────────────────────────
    print("\n[5/9] Cluster quality analysis …")
    _, labels = induce_centers(q_embs_norm, CFG["K"], seed=args.seed)
    soft_pool = compute_teacher_distributions(q_embs_norm, centroids, CFG["temperature"])
    report    = cluster_quality_report(q_embs_norm, labels, centroids, soft_pool)

    with open(os.path.join(args.output_dir, "table2_cluster_analysis.json"), "w") as f:
        json.dump(report, f, indent=2)
    print(f"  Overall micro accuracy: {report['overall']['micro_acc']}%")

    # ── 6. Train CenterDistill ────────────────────────────────────────────────
    if not args.skip_train:
        print("\n[6/9] Training CenterDistill …")
        from datasets import Dataset
        from centerdistill.data import make_tokenise_fn_with_soft_labels

        # Build soft label map: example_id → teacher distribution
        soft_label_map = {}
        for ex, sl in zip(cluster_pool, soft_pool.tolist()):
            soft_label_map[ex["id"]] = sl

        tokenise_fn = make_tokenise_fn_with_soft_labels(
            tokenizer, soft_label_map,
            max_len=CFG["max_len"], stride=CFG["stride"]
        )
        train_dataset = Dataset.from_list(train_en).map(tokenise_fn, batched=False)

        cd_model = CenterDistillModel(bl_dir, CFG["K"], CFG["lambda_kl"])
        train_args = build_training_args(
            output_dir   = cd_dir,
            epochs       = CFG["epochs"],
            batch_size   = CFG["batch_size"],
            grad_accum   = CFG["grad_accum"],
            lr           = CFG["lr"],
            warmup_ratio = CFG["warmup_ratio"],
            weight_decay = CFG["weight_decay"],
            seed         = args.seed,
        )
        trainer = CenterDistillTrainer(
            model=cd_model, args=train_args,
            train_dataset=train_dataset,
        )
        trainer.train()
        trainer.save_model(cd_dir)
        tokenizer.save_pretrained(cd_dir)

        # Patch to HF-compatible format for pipeline()
        patch_to_hf_qa_model(cd_model, bl_dir, tokenizer, cd_dir)
    else:
        print("\n[6/9] Skipping training (--skip_train set)")

    # ── 7. QA evaluation ──────────────────────────────────────────────────────
    print("\n[7/9] QA evaluation …")
    bl_en, _, _ = evaluate_qa(bl_dir, test_en, CFG["n_eval"])
    cd_en, _, _ = evaluate_qa(cd_dir, test_en, CFG["n_eval"])
    bl_es, _, _ = evaluate_qa(bl_dir, test_es, CFG["n_eval_es"])
    cd_es, _, _ = evaluate_qa(cd_dir, test_es, CFG["n_eval_es"])
    bl_de, _, _ = evaluate_qa(bl_dir, test_de, CFG["n_eval_de"])
    cd_de, _, _ = evaluate_qa(cd_dir, test_de, CFG["n_eval_de"])

    print(f"  en-en  BL={bl_en['f1']:.2f}  CD={cd_en['f1']:.2f}")
    print(f"  en-es  BL={bl_es['f1']:.2f}  CD={cd_es['f1']:.2f}")
    print(f"  en-de  BL={bl_de['f1']:.2f}  CD={cd_de['f1']:.2f}")

    # ── 8. Behaviour evaluation ───────────────────────────────────────────────
    print("\n[8/9] Behaviour evaluation (en-es) …")
    from centerdistill.cluster import encode_questions, compute_teacher_distributions
    enc = __import__("sentence_transformers", fromlist=["SentenceTransformer"]).SentenceTransformer(
        "sentence-transformers/LaBSE"
    )

    test_es_sample = test_es[:CFG["n_eval_es"]]
    q_embs_es      = encode_questions([ex["question"] for ex in test_es_sample])
    PT_es          = compute_teacher_distributions(q_embs_es, centroids, CFG["temperature"])

    # Student distributions from trained center_head
    from safetensors.torch import load_file as load_sf
    cd_model_eval = CenterDistillModel(bl_dir, CFG["K"], CFG["lambda_kl"])
    state = load_sf(os.path.join(cd_dir, "model.safetensors"), device="cpu")
    state.pop("mean_soft_labels", None)
    cd_model_eval.load_state_dict(state, strict=False)
    cd_model_eval.eval().cuda()

    PS_list = []
    with torch.no_grad():
        for ex in test_es_sample:
            enc_input = tokenizer(
                ex["question"], ex["context"],
                return_tensors="pt", truncation=True,
                max_length=CFG["max_len"]
            ).to("cuda")
            out   = cd_model_eval(**enc_input)
            ps    = torch.softmax(out["center_logits"], dim=-1)
            PS_list.append(ps.squeeze().cpu().numpy())
    PS_es = np.stack(PS_list)

    beh = evaluate_behaviour(
        PT_es, PS_es,
        CFG["tau_conf"], CFG["tau_ent"], CFG["tau_multi"],
        CFG["K"]
    )
    ci_lo, ci_hi = bootstrap_ci(beh["gold_labels"], beh["pred_labels"])
    err = error_analysis(
        PT_es, PS_es,
        beh["gold_labels"], beh["pred_labels"],
        beh["gold_centres"], beh["pred_centres"],
        CFG["tau_conf"], CFG["tau_ent"],
    )

    print(f"  Behaviour accuracy : {beh['behaviour_acc']}%  "
          f"(95% CI: [{ci_lo}, {ci_hi}])")
    print(f"  Centre accuracy    : {beh['centre_micro_acc']}%")
    print(f"  Worst-cluster F1   : {beh['worst_cluster_f1']}")

    # ── 9. Save all results ────────────────────────────────────────────────────
    print("\n[9/9] Saving all results …")
    all_results = {
        "seed": args.seed,
        "cfg":  {k: v for k, v in CFG.items()
                 if k not in ("output_dir",)},
        "qa": {
            "en_en": {"baseline": bl_en, "centerdistill": cd_en},
            "en_es": {"baseline": bl_es, "centerdistill": cd_es},
            "en_de": {"baseline": bl_de, "centerdistill": cd_de},
        },
        "behaviour":      beh,
        "bootstrap_ci":   {"lo": ci_lo, "hi": ci_hi},
        "error_analysis": {k: v for k, v in err.items() if k != "errors"},
        "cluster":        report,
    }

    with open(os.path.join(args.output_dir, "FINAL_ALL_RESULTS.json"), "w") as f:
        json.dump(
            {k: v.tolist() if isinstance(v, np.ndarray) else v
             for k, v in all_results.items()},
            f, indent=2, default=str
        )

    print(f"\n✅ All results saved to {args.output_dir}/")
    print(f"   Behaviour accuracy : {beh['behaviour_acc']}%")
    print(f"   en-es QA-F1        : {cd_es['f1']:.2f}%")

    # Figures
    if not args.skip_figs:
        print("\nGenerating figures …")
        figs_dir = os.path.join(args.output_dir, "figures")
        os.makedirs(figs_dir, exist_ok=True)
        plot_tsne(q_embs_norm, labels, centroids, figs_dir)
        plot_cluster_summary(report, figs_dir)
        plot_system_performance(
            beh["centre_micro_acc"], beh["behaviour_acc"],
            {"rouge1": 36.5, "rouge2": 22.2, "rougeL": 36.5, "bleu": 7.76},
            cd_es["exact_match"], cd_es["f1"],
            figs_dir,
        )
        plot_silhouette_sweep(derived["silhouette_by_k"], CFG["K"], figs_dir)
        print(f"  ✅ Figures saved to {figs_dir}/")


if __name__ == "__main__":
    main()
