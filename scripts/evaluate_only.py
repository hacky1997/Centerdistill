"""
scripts/evaluate_only.py — Reproduce all paper tables from a saved model.

Usage
-----
    python scripts/evaluate_only.py \
        --output_dir outputs/centerdistill_seed42 \
        --mlqa_root  MLQA_V1

Requires that run_pipeline.py has already been executed (or a saved model
exists at output_dir/centerdistill/).
"""

import os
import sys
import json
import argparse
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from centerdistill.config   import load_config, lock_seed
from centerdistill.data     import load_en_en, load_en_es, load_en_de, evaluate_qa
from centerdistill.cluster  import encode_questions, compute_teacher_distributions
from centerdistill.evaluate import evaluate_behaviour, error_analysis, bootstrap_ci


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--output_dir", required=True)
    p.add_argument("--mlqa_root",  default="MLQA_V1")
    return p.parse_args()


def print_table3(beh, bl_es, cd_es):
    print("\n" + "=" * 72)
    print("TABLE 3 — Baseline Comparison (en-es, N=1000, seed=42)")
    print("=" * 72)
    print(f"{'Method':<30} {'BehAcc':>8} {'WC-F1':>8} {'QA-F1':>8}")
    print("-" * 72)

    rows = [
        ("MLQA Baseline",           "—",    "—",   "74.0"),
        ("AmbigQA",                 "—",    "—",   "71.3"),
        ("Majority-Class (CLARIFY)","75.9%","—",    "—"  ),
        ("Standard XLM-R",          "8.6%", "0.3", f"{bl_es['f1']:.1f}"),
        ("Confidence-based",        "81.4%","7.8",  "—"  ),
        ("Multi-task",              "71.4%","6.5",  "—"  ),
    ]
    for name, ba, wf, qf in rows:
        print(f"  {name:<28} {ba:>8} {wf:>8} {qf:>8}")

    print(f"\n▶ CenterDistill (Ours)         "
          f"{beh['behaviour_acc']:>7.1f}%"
          f"  {beh['worst_cluster_f1']['score']:>7.1f}"
          f"  {cd_es['f1']:>7.1f}")
    print("=" * 72)


def print_table2(report):
    print("\n" + "=" * 62)
    print(f"TABLE 2 — Cluster Analysis (K={report['K']}, seed=42)")
    print("=" * 62)
    print(f"{'Center':<10} {'Size':>6} {'Purity':>10} "
          f"{'Silhouette':>12} {'Acc':>10}")
    print("-" * 62)
    for c in report["centers"]:
        print(f"  Center {c['id']}  {c['size']:>6}  "
              f"{c['purity']:>8.2f}%  {c['silhouette']:>12.2f}  "
              f"{c['model_acc']:>8.2f}%")
    o = report["overall"]
    print(f"  Overall   {o['size']:>6}  {o['purity_mean']:>8.2f}%  "
          f"{o['sil_mean']:>12.2f}  {o['micro_acc']:>8.2f}%")
    print("=" * 62)


def print_confusion_matrix(gold_labels, pred_labels):
    """Full Table 7 — confusion matrix with per-cell counts (Cell 55 logic)."""
    classes = ["ANSWER", "CLARIFY", "ALTERNATIVES"]
    conf    = {g: {p: 0 for p in classes} for g in classes}
    for g, p in zip(gold_labels, pred_labels):
        if g in conf and p in conf[g]:
            conf[g][p] += 1

    print("\n" + "=" * 60)
    print("TABLE 7 — Confusion Matrix")
    print("Gold rows × Predicted columns")
    print("=" * 60)
    print(f"{'':20s} {'ANSWER':>10} {'CLARIFY':>10} {'ALTERNATIVES':>14}")
    print("-" * 60)
    for gold in classes:
        row = conf[gold]
        print(f"{gold:<20s} {row['ANSWER']:>10} "
              f"{row['CLARIFY']:>10} {row['ALTERNATIVES']:>14}")
    print("=" * 60)

    # LaTeX version
    print("\nLaTeX:")
    print(r"\begin{table}[t]")
    print(r"\centering")
    print(r"\caption{Confusion matrix (gold rows $\times$ predicted columns, \texttt{seed=42}).}")
    print(r"\label{tab:confusion}")
    print(r"\begin{tabular}{lccc}")
    print(r"\toprule")
    print(r"\textbf{Gold $\backslash$ Pred} & \textbf{Answer} & \textbf{Clarify} & \textbf{Alternatives} \\")
    print(r"\midrule")
    for gold in classes:
        row = conf[gold]
        print(f"{gold:<12} & {row['ANSWER']:>4} & {row['CLARIFY']:>5} & {row['ALTERNATIVES']:>12} \\\\")
    print(r"\bottomrule")
    print(r"\end{tabular}")
    print(r"\end{table}")

    return conf


def print_error_table(err):
    print("\n" + "=" * 60)
    print("TABLE 8 — Error Breakdown by Category")
    print("=" * 60)
    for cat, n in sorted(err["breakdown"].items(), key=lambda x: -x[1]):
        pct = n / err["n_errors"] * 100
        print(f"  {cat:<27} {n:>4}  ({pct:.1f}% of errors)")
    print("-" * 60)
    print(f"  Total errors: {err['n_errors']} / {err['n_total']} "
          f"({err['n_errors'] / err['n_total'] * 100:.1f}%)")

    if err.get("proximity"):
        print("\n  Threshold proximity (mean distance from decision boundary):")
        for cat, margin in err["proximity"].items():
            print(f"    {cat:<27}: {margin:+.4f}")
    print("=" * 60)


def main():
    args = parse_args()

    cfg_path = os.path.join(args.output_dir, "config.json")
    if not os.path.exists(cfg_path):
        print(f"ERROR: config.json not found at {cfg_path}")
        print("Run scripts/run_pipeline.py first.")
        sys.exit(1)

    CFG = load_config(cfg_path)
    lock_seed(CFG["seed"])

    bl_dir  = os.path.join(args.output_dir, "baseline_en")
    cd_dir  = os.path.join(args.output_dir, "centerdistill")
    centroids = np.load(os.path.join(args.output_dir, "centroids.npy"))

    print("Loading data …")
    _, _, test_en = load_en_en(args.mlqa_root)
    dev_es, test_es = load_en_es(args.mlqa_root)
    dev_de, test_de = load_en_de(args.mlqa_root)

    print("\nQA evaluation …")
    bl_es, _, _ = evaluate_qa(bl_dir, test_es, CFG.get("n_eval_es", 1000))
    cd_es, _, _ = evaluate_qa(cd_dir, test_es, CFG.get("n_eval_es", 1000))
    bl_de, _, _ = evaluate_qa(bl_dir, test_de, CFG.get("n_eval_de", 500))
    cd_de, _, _ = evaluate_qa(cd_dir, test_de, CFG.get("n_eval_de", 500))

    print("\nBehaviour evaluation (en-es) …")
    test_es_sample = test_es[:CFG.get("n_eval_es", 1000)]
    q_embs_es      = encode_questions([ex["question"] for ex in test_es_sample])
    PT_es          = compute_teacher_distributions(
        q_embs_es, centroids, CFG["temperature"]
    )

    import torch
    from transformers import AutoTokenizer
    from centerdistill.model import CenterDistillModel
    from safetensors.torch import load_file as load_sf

    tokenizer     = AutoTokenizer.from_pretrained(bl_dir)
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
                max_length=CFG.get("max_len", 384)
            ).to("cuda")
            out  = cd_model_eval(**enc_input)
            ps   = torch.softmax(out["center_logits"], dim=-1)
            PS_list.append(ps.squeeze().cpu().numpy())
    PS_es = np.stack(PS_list)

    beh = evaluate_behaviour(
        PT_es, PS_es,
        CFG["tau_conf"], CFG["tau_ent"], CFG["tau_multi"], CFG["K"]
    )
    ci_lo, ci_hi = bootstrap_ci(beh["gold_labels"], beh["pred_labels"])
    err = error_analysis(
        PT_es, PS_es,
        beh["gold_labels"], beh["pred_labels"],
        beh["gold_centres"], beh["pred_centres"],
        CFG["tau_conf"], CFG["tau_ent"],
    )

    # Load cluster report if available
    cpath = os.path.join(args.output_dir, "table2_cluster_analysis.json")
    if os.path.exists(cpath):
        with open(cpath) as f:
            report = json.load(f)
        print_table2(report)

    print_table3(beh, bl_es, cd_es)
    print_confusion_matrix(beh["gold_labels"], beh["pred_labels"])
    print_error_table(err)

    print(f"\n  Behaviour accuracy : {beh['behaviour_acc']}%  "
          f"(95% CI [{ci_lo}, {ci_hi}])")
    print(f"  Centre accuracy    : {beh['centre_micro_acc']}%")
    print(f"  en-es QA F1        : {cd_es['f1']:.2f}%")
    print(f"  en-de QA F1        : {cd_de['f1']:.2f}%  (N=500)")


if __name__ == "__main__":
    main()
