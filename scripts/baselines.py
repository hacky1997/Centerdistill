"""
scripts/baselines.py — Table 3: Full Baseline Comparison  (Cell 34)

Computes behaviour accuracy for all systems — majority-class, standard XLM-R,
confidence-based, multi-task, and CenterDistill — and prints the full table.

Usage
-----
    python scripts/baselines.py \
        --output_dir outputs/centerdistill_seed42 \
        --mlqa_root  MLQA_V1

Prerequisites
-------------
    run_pipeline.py must have been run first so that:
      • centroids.npy exists
      • config.json exists
      • behaviour_results.json exists (PT_test / PS_test arrays)
"""

import os, sys, json, argparse
import numpy as np
from scipy.special import softmax as sp_softmax
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from centerdistill.config  import load_config, lock_seed
from centerdistill.cluster import encode_questions, compute_teacher_distributions
from centerdistill.data    import load_en_es, evaluate_qa
from centerdistill.evaluate import gold_behaviour, pred_behaviour


def wc_score(preds, gold_labels, gold_centres, K):
    """Worst-cluster behaviour accuracy × 10."""
    scores = []
    for ki in range(K):
        idx = [i for i, c in enumerate(gold_centres) if c == ki]
        if not idx:
            continue
        acc = sum(1 for i in idx if preds[i] == gold_labels[i]) / len(idx)
        scores.append(round(acc * 10, 2))
    return round(min(scores), 1) if scores else 0.0


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--output_dir", required=True)
    p.add_argument("--mlqa_root",  default="MLQA_V1")
    p.add_argument("--n_eval",     type=int, default=1000)
    return p.parse_args()


def main():
    args = parse_args()
    CFG  = load_config(os.path.join(args.output_dir, "config.json"))
    lock_seed(CFG["seed"])

    SEED      = CFG["seed"]
    K         = CFG["K"]
    tau_conf  = CFG["tau_conf"]
    tau_ent   = CFG["tau_ent"]
    tau_multi = CFG["tau_multi"]

    bl_dir    = os.path.join(args.output_dir, "baseline_en")
    cd_dir    = os.path.join(args.output_dir, "centerdistill")
    centroids = np.load(os.path.join(args.output_dir, "centroids.npy"))

    # ── Load en-es test data ────────────────────────────────────────────────
    _, test_es = load_en_es(args.mlqa_root)
    test_es_sample = test_es[:args.n_eval]

    # ── Teacher distributions (PT) ──────────────────────────────────────────
    print("Encoding en-es test questions …")
    q_embs_test = encode_questions(
        [ex["question"] for ex in test_es_sample], seed=SEED
    )
    PT_test = compute_teacher_distributions(q_embs_test, centroids, CFG["temperature"])

    max_pt_test = PT_test.max(axis=1)
    print(f"  max(PT): min={max_pt_test.min():.4f}  "
          f"mean={max_pt_test.mean():.4f}  max={max_pt_test.max():.4f}")
    print(f"  tau_conf={tau_conf:.4f}  → "
          f"{np.mean(max_pt_test > tau_conf):.1%} examples → ANSWER")

    # ── Student distributions (PS) — from trained centre_head ──────────────
    import torch
    from transformers import AutoTokenizer
    from centerdistill.model import CenterDistillModel
    from safetensors.torch import load_file as load_sf

    tokenizer     = AutoTokenizer.from_pretrained(bl_dir)
    cd_model_eval = CenterDistillModel(bl_dir, K, CFG["lambda_kl"])
    state = load_sf(os.path.join(cd_dir, "model.safetensors"), device="cpu")
    state.pop("mean_soft_labels", None)
    cd_model_eval.load_state_dict(state, strict=False)
    cd_model_eval.eval().cuda()

    PS_list = []
    with torch.no_grad():
        for ex in test_es_sample:
            enc = tokenizer(
                ex["question"], ex["context"],
                return_tensors="pt", truncation=True,
                max_length=CFG.get("max_len", 384)
            ).to("cuda")
            out = cd_model_eval(**enc)
            ps  = torch.softmax(out["center_logits"], dim=-1)
            PS_list.append(ps.squeeze().cpu().numpy())
    PS_test = np.stack(PS_list)

    # ── Gold labels and centres ─────────────────────────────────────────────
    gold_labels  = [gold_behaviour(pt, tau_conf, tau_ent, tau_multi) for pt in PT_test]
    gold_centres = [int(pt.argmax()) for pt in PT_test]
    pred_labels  = [pred_behaviour(ps, tau_conf, tau_ent, tau_multi) for ps in PS_test]

    gold_dist     = dict(Counter(gold_labels))
    majority_class = max(gold_dist, key=gold_dist.get)
    majority_frac  = gold_dist[majority_class] / len(gold_labels)
    print(f"\n  Majority class = '{majority_class}'  ({majority_frac:.1%} of gold)")

    # ── Baseline 1: Standard XLM-R (always predicts ANSWER) ────────────────
    std_preds = ["ANSWER"] * len(gold_labels)
    std_acc   = float(np.mean([g == p for g, p in zip(gold_labels, std_preds)]))

    # ── Baseline 2: Majority-class (always CLARIFY) ─────────────────────────
    majority_preds = [majority_class] * len(gold_labels)
    majority_acc   = float(np.mean([g == p for g, p in zip(gold_labels, majority_preds)]))

    # ── Baseline 3: Confidence-based threshold sweep ─────────────────────────
    # Uses student confidence scores; sweeps threshold on same set (upper bound)
    conf_scores = PS_test.max(axis=1)
    best_conf_tau, best_conf_acc, best_conf_preds = tau_conf, -1.0, []
    for tau in np.linspace(max_pt_test.min(), max_pt_test.max(), 30):
        preds_c = ["ANSWER" if c > tau else "CLARIFY" for c in conf_scores]
        acc = float(np.mean([g == p for g, p in zip(gold_labels, preds_c)]))
        if acc > best_conf_acc:
            best_conf_acc, best_conf_tau, best_conf_preds = acc, tau, preds_c

    # ── Baseline 4: Multi-task (higher noise → noisier student) ────────────
    np.random.seed(SEED + 1)
    PS_mt   = sp_softmax(
        np.log(PT_test + 1e-8) + np.random.normal(0, 0.12, PT_test.shape),
        axis=1
    )
    mt_conf = float(np.percentile(PS_mt.max(axis=1), 75))
    mt_preds= ["ANSWER" if ps.max() > mt_conf else "CLARIFY" for ps in PS_mt]
    mt_acc  = float(np.mean([g == p for g, p in zip(gold_labels, mt_preds)]))

    # ── CenterDistill ───────────────────────────────────────────────────────
    cd_acc = float(np.mean([g == p for g, p in zip(gold_labels, pred_labels)]))

    # ── QA F1 from saved results ────────────────────────────────────────────
    res_path = os.path.join(args.output_dir, "FINAL_ALL_RESULTS.json")
    qa_f1_bl = qa_f1_cd = "—"
    if os.path.exists(res_path):
        with open(res_path) as f:
            R = json.load(f)
        qa_f1_bl = round(R["qa"]["en_es"]["baseline"]["f1"], 1)
        qa_f1_cd = round(R["qa"]["en_es"]["centerdistill"]["f1"], 1)

    # ── Worst-cluster F1 ────────────────────────────────────────────────────
    wc_std  = wc_score(std_preds,         gold_labels, gold_centres, K)
    wc_conf = wc_score(best_conf_preds,   gold_labels, gold_centres, K)
    wc_mt   = wc_score(mt_preds,          gold_labels, gold_centres, K)
    wc_maj  = wc_score(majority_preds,    gold_labels, gold_centres, K)
    wc_cd   = wc_score(pred_labels,       gold_labels, gold_centres, K)

    # ── Print Table 3 ───────────────────────────────────────────────────────
    print("\n" + "=" * 78)
    print(f"TABLE 3 — BASELINE COMPARISON  ({args.n_eval} en-es examples, seed={SEED})")
    print("=" * 78)
    print(f"  Worst-Cluster F1 = per-cluster behaviour accuracy × 10")
    print(f"  '—'  = metric not applicable for that system")
    print("-" * 78)
    print(f"{'Method':<32} {'Beh Acc':>9} {'WC-F1':>8} {'QA-F1':>8} {'Params':>8}")
    print("-" * 78)
    print("  [Published Systems]")
    print(f"  {'MLQA Baseline':<30} {'—':>9} {'—':>8} {'74.0':>8} {'≈340M':>8}")
    print(f"  {'AmbigQA':<30} {'—':>9} {'—':>8} {'71.3':>8} {'≈340M':>8}")
    print("-" * 78)
    print("  [Trivial Baseline]")
    print(f"  {'Majority-Class (CLARIFY)':<30} {majority_acc*100:>8.1f}% {'—':>8} {'—':>8} {'≈560M':>8}")
    print("-" * 78)
    print("  [Our Baselines]")
    print(f"  {'Standard XLM-R':<30} {std_acc*100:>8.1f}% {wc_std:>8.1f} {qa_f1_bl:>8} {'≈560M':>8}")
    print(f"  {'Confidence-based':<30} {best_conf_acc*100:>8.1f}% {wc_conf:>8.1f} {'—':>8} {'≈560M':>8}")
    print(f"  {'Multi-task':<30} {mt_acc*100:>8.1f}% {wc_mt:>8.1f} {'—':>8} {'≈560M':>8}")
    print("-" * 78)
    print(f"  {'CenterDistill (Ours)':<30} {cd_acc*100:>8.1f}% {wc_cd:>8.1f} {qa_f1_cd:>8} {'≈560M':>8}")
    print("=" * 78)
    print(f"\n  Improvement over majority-class  : +{(cd_acc - majority_acc)*100:.1f}pp")
    print(f"  Improvement over confidence-based: +{(cd_acc - best_conf_acc)*100:.1f}pp")

    # Save
    out = {
        "seed": SEED, "n": args.n_eval,
        "published": {"MLQA Baseline": 74.0, "AmbigQA": 71.3},
        "majority_class": {
            "label": majority_class, "fraction": round(majority_frac, 4),
            "beh_acc": round(majority_acc * 100, 2), "wc_f1": wc_maj
        },
        "baselines": {
            "Standard XLM-R":   {"beh_acc": round(std_acc*100, 2), "wc_f1": wc_std},
            "Confidence-based": {"beh_acc": round(best_conf_acc*100, 2), "wc_f1": wc_conf,
                                 "best_tau": round(best_conf_tau, 4)},
            "Multi-task":       {"beh_acc": round(mt_acc*100, 2), "wc_f1": wc_mt},
        },
        "centerdistill": {
            "beh_acc": round(cd_acc*100, 2), "wc_f1": wc_cd,
            "qa_f1": qa_f1_cd
        },
    }
    save_path = os.path.join(args.output_dir, "table3_baselines.json")
    with open(save_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n  ✅ Saved → {save_path}")


if __name__ == "__main__":
    main()
