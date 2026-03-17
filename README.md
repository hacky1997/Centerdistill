<div align="center">

# CenterDistill: Weakly-Supervised Distillation for Ambiguity-Aware Cross-Lingual QA

### Weakly-Supervised Distillation for Ambiguity-Aware Cross-Lingual Question Answering

[![EAAAI 2026](https://img.shields.io/badge/EAAAI-2026-4B6FA8?style=for-the-badge)](https://eaaai.org)
[![Python 3.9+](https://img.shields.io/badge/Python-3.9%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![License: Apache2](https://img.shields.io/badge/license-Apache%202-blue?style=for-the-badge)](LICENSE)
[![Open in Colab](https://img.shields.io/badge/Open%20in-Colab-F9AB00?style=for-the-badge&logo=googlecolab&logoColor=white)](https://colab.research.google.com/github/hacky1997/centerdistill/blob/main/notebooks/CenterDistill_Colab.ipynb)
[![arXiv](https://img.shields.io/badge/)](https://arxiv.org/abs/2504.17894)
[![DOI](https://img.shields.io/badge/))](https://doi.org/XXXX.XXXX)

**Somyajit Chakraborty¹ · Sayak Naskar² · Soham Paul² · Angshuman Jana² · Nilotpal Chakraborty² · Avijit Gayen²·³**

¹ University College Cork &nbsp;|&nbsp; ² IIIT Guwahati &nbsp;|&nbsp; ³ Techno India University

</div>

---

## The Problem

QA systems return a single answer even when a question is genuinely ambiguous — when it could mean different things depending on context, or when multiple valid answers exist across different interpretations. This forces users to rephrase, re-query, or accept a potentially wrong answer.

**CenterDistill** solves this by giving a model three behavioural choices at inference time:

| Signal | Behaviour |
|--------|-----------|
| High confidence: max P(c\|q) > τ | **Answer** directly |
| High entropy: H(P) > τ | **Clarify** — ask the user what they meant |
| Bimodal mass: two centres > τ | **Return Alternatives** — provide both answers |

No ambiguity labels are required. The training signal comes entirely from the geometry of question embeddings.

---

## Workflow Diagram
![CenterDistill_Pipeline](CenterDistill%20Workflow%20Pipeline.png)

---
## How It Works

```
Question + Context
        │
   XLM-RoBERTa-large         (shared encoder — frozen/fine-tuned)
        │
   ┌────┴──────────────────────────┐
   │                               │
span_head                     center_head
Linear(hidden → 2)            Linear(hidden → K)
start / end logits            centre distribution P_S(c|q,d)
   │                               │
Extractive answer             Behaviour policy
                         ┌─────────┼────────────────┐
                    max(PS)>τ   H(PS)>τ_ent    two peaks>τ_multi
                         │          │                │
                      ANSWER    CLARIFY        ALTERNATIVES
```

**Training objective** — a convex combination of span extraction loss and KL distillation from a soft teacher:

```
L = λ · KL( P_T(c|q) ‖ P_S(c|q,d) )  +  (1−λ) · L_span(start*, end*)
```

The teacher distribution P_T is computed offline using LaBSE embeddings and spectral clustering — no annotation required. All six hyperparameters are derived automatically from the training data before any gradient step.

---

## Results

### Table 3 — Behaviour Selection (en–es, N=1,000, seed=42)

| Method | Beh. Acc. | WC-F1 | QA-F1 | Params |
|:-------|:---------:|:-----:|:-----:|:------:|
| MLQA Baseline (Lewis et al., 2019) | — | — | 74.0 | ≈340M |
| AmbigQA (Min et al., 2020) | — | — | 71.3 | ≈340M |
| Majority-Class (always Clarify) | 75.9% | — | — | ≈560M |
| Standard XLM-R | 8.6% | 0.3 | 77.3 | ≈560M |
| Confidence-based threshold | 81.4% | 7.8 | — | ≈560M |
| Multi-task distillation | 71.4% | 6.5 | — | ≈560M |
| **CenterDistill (Ours)** | **90.1%** | **8.8** | **77.3** | ≈560M |

95% bootstrap confidence interval: **[88.2%, 91.8%]**  
WC-F1 = Worst-Cluster F1 (per-cluster behaviour accuracy × 10, lower bound on robustness)

### Table 5 — Cross-Lingual Transfer

| Pair | N | Baseline F1 | CD F1 | Beh. Acc. | WC-F1 |
|:-----|:-:|:-----------:|:-----:|:---------:|:-----:|
| en–es | 1,000 | 77.3 | 75.2 | 90.1% | 8.8 |
| en–de | 500 | 75.6 | 74.6 | 91.0% | 8.7 |

---

## Repository Structure

```
centerdistill/
│
├── centerdistill/                  Importable Python package
│   ├── __init__.py
│   ├── config.py                   BASE_CFG + derive_hyperparameters()
│   │                               Cells 3a, 3b → K, λ, τ, temperature, thresholds
│   │
│   ├── data.py                     MLQA loaders, tokenisation, QA evaluation
│   │                               Cell 4  → load_en_en/es/de, make_tokenise_fn
│   │                               Cell 6  → evaluate_qa()
│   │
│   ├── cluster.py                  Centre induction + teacher distributions
│   │                               Cell 7  → encode_questions() (LaBSE)
│   │                               Cell 8  → induce_centers(), cluster_quality_report()
│   │                               Cells 12,14 → compute_teacher_distributions()
│   │
│   ├── model.py                    Architecture + training
│   │                               Cell 9  → CenterDistillModel, CenterDistillTrainer
│   │                               Cell 9b → patch_to_hf_qa_model()
│   │
│   ├── evaluate.py                 Evaluation and analysis
│   │                               Cell 12 → evaluate_behaviour(), gold/pred_behaviour()
│   │                               Cell 13 → error_analysis(), bootstrap_ci()
│   │
│   └── visualize.py                All publication figures (300–900 dpi)
│                                   Cell 15 → plot_system_performance()      (Figure 6)
│                                   Cell 16 → plot_metrics_heatmap()         (Figure 7)
│                                   Cell 17 → plot_silhouette_sweep()        (Figure S1)
│                                   Cells 18,18b → plot_k_ablation()        (Figure S2)
│                                   Cell 8  → plot_tsne(), plot_cluster_summary() (Figs 4,5)
│
├── scripts/
│   ├── run_pipeline.py             End-to-end: download → embed → train → eval → save
│   │                               Covers: Cells 2,5,6,7,10,11,11b,12,20
│   │
│   ├── evaluate_only.py            Reproduce all tables from a saved model
│   │                               Covers: Cells 2 (plot setup), 6 (confusion matrix + tables)
│   │
│   ├── baselines.py                Table 3 — all baseline comparisons
│   │                               Covers: Cell 13 exactly
│   │
│   ├── ablation.py                 Table 4 — K ablation sweep
│   │                               Covers: Cell 14 exactly + entropy diagnostics
│   │
│   ├── generate_latex.py           LaTeX source for Tables 2–5 + confusion matrix
│   │                               Covers: Cell 19 exactly
│   │
│   └── check_leakage.py            Data leakage verification (ID + context + question)
│                                   Covers: Cell data-leakage-check exactly
│
├── notebooks/
│   └── CenterDistill_Colab.ipynb   Original Colab notebook (all 28 cells)
│
├── results/                        Saved JSON artefacts (populated at runtime)
├── figures/                        Generated figures (populated at runtime)
├── requirements.txt                Pinned dependencies
├── setup.py                        Pip-installable package
└── LICENSE                         MIT
```
---

## Hyperparameters

All six data-driven parameters are derived automatically in `config.derive_hyperparameters()` before any model training begins. No manual tuning is required or performed.

| Parameter | Value | Derivation method |
|:----------|:-----:|:------------------|
| **K** (centres) | 5 | Silhouette sweep K ∈ {2…8} with cosine affinity; K=5 maximises score among semantically meaningful partitions (K ≥ 4) |
| **τ** (temperature) | 10.0 | Maximises discrimination score: std(max P_T) × above-chance mass × (1 − concentration penalty) |
| **λ** (KL weight) | 0.7 | Information-theoretic ideal ≈ 0.91; capped at 0.7 to protect extractive QA quality |
| **τ\_conf** | 0.44 | 75th percentile of max(P_T) on training pool; guarantees ≥ 20% routed to ANSWER |
| **τ\_ent** | 1.51 | Median entropy of training examples not routed to ANSWER |
| **τ\_multi** | 0.24 | 60th percentile of the second-highest P_T mass |

> **Note on K=6/7 ablation numbers:** At K ≥ 6, mean teacher entropy drops below 0.8 nats, making soft labels near-deterministic. The reported 98.5% and 99.1% behaviour accuracy reflect label sharpness rather than better semantic structure — silhouette scores confirm the underlying geometry does not improve beyond K=5.

---

## Quick Start

### Option A — Google Colab (recommended)

Click the badge at the top. Run all cells top-to-bottom. The full pipeline — MLQA download, hyperparameter derivation, training, evaluation, and all figures — completes in approximately **45 minutes on a T4 GPU**.

### Option B — Local / Slurm

**1. Install**
```bash
git clone https://github.com/YOUR_USERNAME/centerdistill.git
cd centerdistill
pip install -r requirements.txt
# or: pip install -e .
```

**2. Download MLQA**
```bash
curl -L -o MLQA_V1.zip https://dl.fbaipublicfiles.com/MLQA/MLQA_V1.zip
unzip -q MLQA_V1.zip && rm MLQA_V1.zip
```

**3. Train and evaluate end-to-end**
```bash
python scripts/run_pipeline.py \
    --mlqa_root  MLQA_V1 \
    --output_dir outputs/centerdistill_seed42 \
    --seed       42
```

**4. Reproduce individual tables from a saved model**
```bash
# All tables + confusion matrix
python scripts/evaluate_only.py  --output_dir outputs/centerdistill_seed42 --mlqa_root MLQA_V1

# Table 3: baseline comparison
python scripts/baselines.py      --output_dir outputs/centerdistill_seed42 --mlqa_root MLQA_V1

# Table 4: K ablation
python scripts/ablation.py       --output_dir outputs/centerdistill_seed42 --mlqa_root MLQA_V1

# LaTeX source for all tables
python scripts/generate_latex.py --output_dir outputs/centerdistill_seed42

# Data leakage verification
python scripts/check_leakage.py  --mlqa_root MLQA_V1
```

---

## Reproducibility

All experiments use `seed=42` locked across Python `random`, NumPy, PyTorch, and HuggingFace `set_seed`.

| Check | Result |
|:------|:------:|
| Train / test ID overlap | 0 |
| Cluster pool / test ID overlap | 0 |
| Context text overlap (pool vs any test set) | 0 |
| Question text overlap (train vs test) | 1 (documented; removing it changes no metric) |
| Spectral clustering stability (n_init=5) | Identical assignments across all initialisations |
| Single Colab T4 run reproduces all Table 2–5 numbers exactly | ✅ |

Run `python scripts/check_leakage.py` to verify all of the above on your machine.

---

## Citation

```bibtex
@inproceedings{chakraborty2026centerdistill,
  title     = {CenterDistill: Weakly-Supervised Distillation for
               Ambiguity-Aware Cross-Lingual Question Answering},
  author    = {Chakraborty, Somyajit and Naskar, Sayak and Paul, Soham
               and Jana, Angshuman and Chakraborty, Nilotpal and Gayen, Avijit},
  booktitle = {CenterDistill: Weakly-Supervised Distillation for Ambiguity-Aware Cross-Lingual QA},
  year      = {2026}
}
```

---

<div align="center">
Distributed under the Apache 2.0 License. See <a href="LICENSE">License</a> for more information. 
</div>
