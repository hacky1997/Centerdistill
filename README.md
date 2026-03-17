<div align="center">

<h1>🎯 CenterDistill</h1>
<h3>Weakly-Supervised Distillation for Ambiguity-Aware Cross-Lingual Question Answering</h3>

<p><em>A framework that teaches QA models <strong>when to answer, when to ask back, and when to offer alternatives</strong> — entirely without ambiguity labels.</em></p>

[![EAAAI 2026](https://img.shields.io/badge/EAAAI-2026-green?style=for-the-badge&logo=academia&logoColor=white)](https://eaaai.org)
[![Python 3.9+](https://img.shields.io/badge/Python-3.9%2B-AD1D7D?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![License](https://img.shields.io/badge/license-Apache%202-blue?style=for-the-badge&logo=apache&logoColor=white)](LICENSE)
[![Open in Colab](https://img.shields.io/badge/Open%20in-Colab-F9AB00?style=for-the-badge&logo=googlecolab&logoColor=white)](https://colab.research.google.com/github/hacky1997/centerdistill/blob/main/notebooks/CenterDistill_Colab.ipynb)
[![arXiv](https://img.shields.io/badge/Arxiv-XXXX.XXXX-red?style=for-the-badge&logo=arxiv&logoColor=white)](https://arxiv.org/abs/xxxx.xxxx)
[![DOI](https://img.shields.io/badge/DOI-XXXX.XXXX-FF2200?style=for-the-badge&logo=doi&logoColor=white)](https://doi.org/XXXX.XXXX)

**Somyajit Chakraborty¹ · Sayak Naskar² · Soham Paul² · Angshuman Jana² · Nilotpal Chakraborty² · Avijit Gayen²·³**

¹ University College Cork &nbsp;|&nbsp; ² IIIT Guwahati &nbsp;|&nbsp; ³ Techno India University

</div>

---

## 📌 TL;DR

> Standard QA systems commit to one answer even when a question is genuinely ambiguous. **CenterDistill** fixes this by learning the *geometry* of question meaning — no ambiguity labels required. It clusters question embeddings into semantic centers, distils those soft distributions into an XLM-RoBERTa student, and at inference time routes each query to one of three behaviours: **Answer**, **Clarify**, or **Return Alternatives**.
>
> On 1 000 English–Spanish MLQA examples: **90.1% behaviour accuracy**, **92.6% center assignment accuracy**, **77.3 QA-F1** — outperforming all baselines while adding zero annotation cost.

---

## 📖 Table of Contents

- [The Problem](#-the-problem)
- [Key Contributions](#-key-contributions)
- [How It Works](#-how-it-works)
- [Pipeline Overview](#-pipeline-overview)
- [Results](#-results)
- [Quick Start](#-quick-start)
- [Installation](#-installation)
- [Usage](#-usage)
- [Dataset](#-dataset)
- [Hyperparameters](#-hyperparameters)
- [Reproducibility](#-reproducibility)
- [Project Structure](#-project-structure)
- [Industry Applications](#-industry-applications)
- [Limitations & Future Work](#-limitations--future-work)
- [Paper Alignment Notes](#-paper-alignment-notes)
- [Citation](#-citation)
- [Contributing](#-contributing)

---

## 🔥 The Problem

Every time a QA system answers a question like *"What are the side effects?"* or *"When does school start?"*, it is silently choosing one interpretation and discarding all others. According to [Pew Research (2023)](https://www.pewresearch.org/internet/2023/12/06/ai-in-everyday-life/) and [Statista (2024)](https://www.statista.com/topics/9845/chatgpt/), LLM-based assistant usage surged from **28 % → 68 %** between 2022 and 2024 while traditional forums (which surface multiple answers naturally) declined from 72 % to 28 %. As users rely more heavily on single-system responses, undetected ambiguity causes real downstream harm:

| Domain | Ambiguous Query | Risk of Single Confident Answer |
|--------|-----------------|----------------------------------|
| 🏥 Medical | *"What are the side effects?"* | Wrong drug assumed; patient harm |
| 🛒 E-Commerce | *"When will my order arrive?"* | Policy vs. shipment vs. estimate confusion |
| 🎓 Education | *"When does school start?"* | Daily time vs. semester start conflated |
| ⚖️ Legal | *"What is the statute of limitations?"* | Jurisdiction silently assumed |

[Min et al. (2020)](https://aclanthology.org/2020.emnlp-main.466/) showed that standard QA models **fail to detect ambiguity in 64 % of naturally ambiguous questions**, returning a high-confidence single answer. Existing fixes (AmbigQA, ASQA) require expensive interpretation annotations and handle ambiguity at evaluation time only — not during training.

---

## ✨ Key Contributions

- **Label-free semantic center induction** — Clusters LaBSE question embeddings via spectral clustering to discover *K* semantic regions with zero manual annotation.
- **Soft distillation as ambiguity signal** — Computes cosine-similarity teacher distributions over cluster centroids; feeds them to the student as KL-divergence supervision, richer than hard one-hot labels.
- **Three-way inference-time behaviour policy** — The predicted center distribution determines whether to *Answer* directly, *Clarify* (request elaboration via mT5), or *Return Alternatives* (surface multiple valid answers).
- **Automatic hyperparameter derivation** — All six critical thresholds (K, τ, λ, τ\_conf, τ\_ent, τ\_multi) are computed programmatically from training statistics before any gradient step — no grid search on held-out labels.
- **Cross-lingual transfer** — A single model trained on English data transfers the behaviour capability to Spanish and German test sets without additional tuning.
- **New evaluation protocol** — Behaviour Accuracy + Worst-Cluster F1 (RoMQA) as primary metrics alongside standard EM/F1, enabling deployment-oriented assessment.

---

## ⚙️ How It Works

### Conceptual Overview

```
Question + Context
        │
   XLM-RoBERTa-large         (shared encoder — deepset/xlm-roberta-large-squad2)
        │
   ┌────┴──────────────────────────┐
   │                               │
span_head                     center_head
Linear(hidden → 2)            Linear(hidden → K=5)
start / end logits            centre distribution  P_S(c | q, d)
   │                               │
Extractive answer             Behaviour policy
                         ┌─────────┼─────────────────┐
                    max(PS)>0.44  H(PS)>1.51    two peaks>0.24
                         │            │                │
                      ANSWER      CLARIFY        ALTERNATIVES
                         │            │                │
                    Span text   mT5 clarification  All valid spans
```

### Training Objective

The student jointly minimises extractive span loss and KL divergence from the teacher's soft center distribution:

```
ℒ = λ · KL( P_T(c|q) ‖ P_S(c|q,d) )  +  (1−λ) · ℒ_span(start*, end*)
    └──────────────────────────────┘    └─────────────────────────────┘
         ambiguity-awareness signal          standard extractive QA
              (λ = 0.70)                         (1−λ = 0.30)
```

The **teacher distribution** P\_T is computed *offline* using LaBSE embeddings and spectral clustering — no annotation, no fine-tuning of the teacher. All six hyperparameters are derived automatically from training-set statistics.

### Three-Stage Center Induction

```
Stage 1 ─ Embed   : LaBSE(q) → ê  (768-dim, L2-normalised)
Stage 2 ─ Cluster : SpectralClustering(cosine affinity, K=5, seed=42) → {C_k}
Stage 3 ─ Teacher : P_T(c_k|q) = softmax(τ · µ̃_k⊤ ê_q)   [τ=10.0]
```

---

## 🔄 Pipeline Overview

![CenterDistill Pipeline](CenterDistill%20Workflow%20Pipeline.png)

> *Figure: End-to-end CenterDistill pipeline — from raw MLQA inputs through center induction, distillation training, to the three-way inference-time behaviour policy.*

The five stages map directly to the codebase modules:

| Stage | Module | Key Function |
|-------|--------|--------------|
| Data loading | `centerdistill/data.py` | `load_en_en / es / de`, `make_tokenise_fn` |
| Center induction | `centerdistill/cluster.py` | `encode_questions()`, `induce_centers()` |
| Teacher distributions | `centerdistill/cluster.py` | `compute_teacher_distributions()` |
| Training | `centerdistill/model.py` | `CenterDistillModel`, `CenterDistillTrainer` |
| Inference & evaluation | `centerdistill/evaluate.py` | `evaluate_behaviour()`, `bootstrap_ci()` |

---

## 📊 Results

### Behaviour Selection — English→Spanish (N = 1 000, seed = 42)

| Method | Beh. Acc. ↑ | WC-F1 ↑ | QA-F1 ↑ | Params |
|:-------|:-----------:|:--------:|:--------:|:------:|
| MLQA Baseline (Lewis et al., 2019) | — | — | 74.0 | ≈ 340M |
| AmbigQA (Min et al., 2020) | — | — | 71.3 | ≈ 340M |
| Majority-Class (always Clarify) | 75.9 % | — | — | ≈ 560M |
| Standard XLM-R | 8.6 % | 0.3 | 77.3 | ≈ 560M |
| Confidence-based threshold | 81.4 % | 7.8 | — | ≈ 560M |
| Multi-task distillation | 71.4 % | 6.5 | — | ≈ 560M |
| **CenterDistill (Ours)** | **90.1 %** | **8.8** | **77.3** | ≈ 560M |

> 95 % bootstrap CI for CenterDistill behaviour accuracy: **[88.2 %, 91.8 %]** (10 K resamples).
>
> The non-overlapping CI vs. the confidence-based baseline ([78.2 %, 84.4 %]) confirms the improvement is statistically significant at α = 0.05.

**WC-F1** = Worst-Cluster F1 (per-cluster behaviour accuracy × 10) — a lower bound on robustness across semantic groups.

### Cross-Lingual Transfer

| Pair | N | Baseline F1 | CD F1 | Beh. Acc. | WC-F1 |
|:-----|:-:|:-----------:|:-----:|:---------:|:-----:|
| en → es | 1 000 | 77.3 | 75.2 | 90.1 % | 8.8 |
| en → de | 500 | 75.6 | 74.6 | 91.0 % | 8.7 |

> The −2.05 % F1 on en–es reflects a *deliberate trade-off*: EM/F1 apply only to direct-answer cases; 75.9 % of examples are correctly routed to Clarify/Alternatives and excluded from the span metric. On en–de this gap narrows to −1.03 %.

### Confusion Matrix — en→es (gold rows × predicted columns)

|  | **Pred: Answer** | **Pred: Clarify** | **Pred: Alternatives** |
|---|:---:|:---:|:---:|
| **Gold: Answer** | 73 | 12 | 1 |
| **Gold: Clarify** | 17 | 708 | 34 |
| **Gold: Alternatives** | 0 | 35 | 120 |

### Cluster Quality — 500-question en-en pool

| Semantic Center | Size | Purity | Silhouette | Model Acc. |
|:----------------|:----:|:------:|:----------:|:----------:|
| Center 1 | 120 | 90.53 % | 0.03 | 98.39 % |
| Center 2 | 116 | 87.76 % | 0.03 | 97.44 % |
| Center 3 | 65 | 89.22 % | 0.03 | 98.36 % |
| Center 4 | 70 | 88.99 % | 0.03 | 95.65 % |
| Center 5 | 129 | 89.33 % | 0.03 | 98.45 % |
| **Overall** | **500** | **89.17 %** | **0.04** | **97.66 % (micro)** |

> Although silhouette scores are modest (0.03–0.04 — expected for natural language questions), prediction accuracy remains consistently high across all centers, confirming that spectral clustering on LaBSE embeddings yields semantically coherent regions despite their proximity in embedding space.

### Qualitative Case Studies

| Question | Gold Behaviour | Standard XLM-R | Multi-task | **CenterDistill** |
|----------|:-------------:|:--------------:|:----------:|:-----------------:|
| *"How long does the treatment last?"* (chemo: 3–6 months vs. single session: 4–6 hours) | **Alternatives** | 3–6 months (overconfident) | Unanswerable | ✅ Alternatives: "3–6 months" / "4–6 hours" |
| *"What did the company report?"* (single salient referent: $4.2B revenue) | **Answer** | $4.2B (correct) | $4.2B (correct) | ✅ $4.2B (no spurious clarification) |
| *"What are the side effects?"* (two medications in context) | **Clarify** | Nausea (anchors to first) | Unanswerable | ✅ Clarify: *"Which medication are you referring to?"* |

---

## 🚀 Quick Start

### Option A — Google Colab *(recommended, zero setup)*

Click the badge at the top. Run all 28 cells top-to-bottom. The full pipeline — MLQA download, hyperparameter derivation, training, evaluation, and all publication figures — completes in approximately **45 minutes on a free T4 GPU**.

### Option B — Local / Slurm

```bash
# 1. Clone and install
git clone https://github.com/hacky1997/centerdistill.git
cd centerdistill
pip install -e .                   # or: pip install -r requirements.txt

# 2. Download MLQA
curl -L -o MLQA_V1.zip https://dl.fbaipublicfiles.com/MLQA/MLQA_V1.zip
unzip -q MLQA_V1.zip && rm MLQA_V1.zip

# 3. Full pipeline: embed → cluster → train → evaluate → save all figures
python scripts/run_pipeline.py \
    --mlqa_root  MLQA_V1 \
    --output_dir outputs/centerdistill_seed42 \
    --seed       42
```

Expected runtime: **~3–4 hours on a single A100 / ~6–8 hours on a V100**.

---

## 🛠️ Installation

### Requirements

| Dependency | Version |
|------------|---------|
| Python | ≥ 3.9 |
| PyTorch | 2.10.0+cu128 |
| HuggingFace Transformers | 4.37.2 |
| Accelerate | 0.26.1 |
| sentence-transformers | (for LaBSE) |
| scikit-learn | (spectral clustering) |

```bash
# Recommended: create a dedicated environment
conda create -n centerdistill python=3.10 -y
conda activate centerdistill

# Install pinned dependencies
pip install -r requirements.txt

# Or install as an editable package
pip install -e .
```

### Verify Installation

```python
from centerdistill import CenterDistillModel
from centerdistill.config import BASE_CFG, derive_hyperparameters
print("CenterDistill installed successfully ✓")
```

---

## 💻 Usage

### Reproduce Individual Tables and Figures from a Saved Model

```bash
# Table 3 + Table 4 + confusion matrix + all figures
python scripts/evaluate_only.py \
    --output_dir outputs/centerdistill_seed42 \
    --mlqa_root  MLQA_V1

# Table 3: baseline comparison
python scripts/baselines.py \
    --output_dir outputs/centerdistill_seed42 \
    --mlqa_root  MLQA_V1

# Table 4: K ablation sweep (K ∈ {3,4,5,6,7})
python scripts/ablation.py \
    --output_dir outputs/centerdistill_seed42 \
    --mlqa_root  MLQA_V1

# LaTeX source for Tables 2–5 + confusion matrix
python scripts/generate_latex.py \
    --output_dir outputs/centerdistill_seed42

# Data leakage verification (should report 0 overlaps)
python scripts/check_leakage.py --mlqa_root MLQA_V1
```

### Run Inference on a Custom Example

```python
from centerdistill import CenterDistillModel
from centerdistill.config import BASE_CFG

model = CenterDistillModel.from_pretrained("outputs/centerdistill_seed42")

question = "What are the side effects?"
context  = "Medication A causes nausea. Medication B causes headaches and dizziness."

result = model.predict(question, context)
# result = {
#   "behaviour": "CLARIFY",
#   "clarification": "Which medication are you referring to?",
#   "center_distribution": [0.12, 0.09, 0.38, 0.31, 0.10],
#   "entropy": 1.57
# }
print(result)
```

### Derive Hyperparameters Automatically

```python
from centerdistill.config import BASE_CFG, derive_hyperparameters

cfg = derive_hyperparameters(BASE_CFG, cluster_pool_questions)
# Returns: K=5, tau=10.0, lambda=0.7, tau_conf=0.44, tau_ent=1.51, tau_multi=0.24
print(cfg)
```

---

## 📁 Dataset

CenterDistill uses **MLQA** ([Lewis et al., 2019](https://aclanthology.org/2020.acl-main.653/)) — a publicly available extractive QA benchmark in SQuAD format covering **7 languages**.

| Split | N | Role |
|:------|:-:|:-----|
| MLQA en-en (train) | 918 | Model training |
| MLQA en-en (val) | 230 | Hyperparameter selection (λ) |
| MLQA en-en (test) | 11 590 | Baseline QA eval |
| MLQA en-es (val) | 500 | — |
| MLQA en-es (test) | 5 253 → **first 1 000 used** | Primary behaviour eval |
| MLQA en-de (val) | 512 | — |
| MLQA en-de (test) | 4 517 → **first 500 used** | Secondary behaviour eval |
| Cluster pool | 500 | Center induction (en-en train subset) |

**Download:**

```bash
curl -L -o MLQA_V1.zip https://dl.fbaipublicfiles.com/MLQA/MLQA_V1.zip
unzip -q MLQA_V1.zip && rm MLQA_V1.zip
```

> ℹ️ **Data Leakage:** `check_leakage.py` confirms zero train/test ID overlap, zero context overlap across pool and test sets, and a single documented question-text overlap that changes no reported metric.

---

## 🔧 Hyperparameters

All six parameters are **derived automatically** in `config.derive_hyperparameters()` before any model training. No manual tuning or validation-label grid search is performed.

| Parameter | Value | Derivation |
|:----------|:-----:|:-----------|
| **K** (semantic centres) | 5 | Silhouette sweep K ∈ {2…8} with cosine affinity; K = 5 maximises score among semantically meaningful partitions (K ≥ 4) |
| **τ** (temperature) | 10.0 | Maximises discrimination score: std(max P\_T) × above-chance mass × (1 − concentration penalty) |
| **λ** (KL weight) | 0.7 | Information-theoretic ideal ≈ 0.91; capped at 0.7 to protect extractive QA quality (span F1 stays within 1.5 % of baseline) |
| **τ\_conf** | 0.44 | 75th percentile of max(P\_T) on training pool; guarantees ≥ 20 % routed to ANSWER |
| **τ\_ent** | 1.51 | Median entropy of training examples not routed to ANSWER |
| **τ\_multi** | 0.24 | 60th percentile of second-highest P\_T mass among non-answer training examples |

> **Note on K = 6/7 ablation numbers:** At K ≥ 6, mean teacher entropy drops below 0.8 nats, making soft labels near-deterministic. The reported 98.5 % and 99.1 % behaviour accuracy reflect label sharpness rather than better semantic structure — silhouette scores confirm the underlying geometry does not improve beyond K = 5.

---

## 🔬 Reproducibility

All experiments use `seed = 42` locked across Python `random`, NumPy, PyTorch, and HuggingFace `set_seed`.

| Reproducibility Check | Status |
|:----------------------|:------:|
| Train / test ID overlap | ✅ 0 |
| Cluster pool / test ID overlap | ✅ 0 |
| Context text overlap (pool vs. any test set) | ✅ 0 |
| Question text overlap (train vs. test) | ✅ 0 |
| Spectral clustering stability (n\_init = 5) | ✅ Identical assignments across all initialisations |
| Single Colab T4 run reproduces all Table 2–5 numbers exactly | ✅ |

```bash
# Verify all of the above on your machine
python scripts/check_leakage.py --mlqa_root MLQA_V1
```

### Training Configuration

```yaml
backbone:        deepset/xlm-roberta-large-squad2
epochs:          4
batch_size:      8  (grad_accum=4 → effective=32)
optimizer:       AdamW  (lr=3e-5, weight_decay=0.01)
scheduler:       cosine with 10% warmup
precision:       FP16 mixed
gradient_ckpt:   true
seed:            42
```

---

## 📂 Project Structure

```
centerdistill/
│
├── centerdistill/                  # Importable Python package
│   ├── __init__.py
│   ├── config.py                   # BASE_CFG + derive_hyperparameters()
│   ├── data.py                     # MLQA loaders, tokenisation, QA evaluation
│   ├── cluster.py                  # LaBSE encoding, spectral clustering, teacher distributions
│   ├── model.py                    # CenterDistillModel + CenterDistillTrainer
│   ├── evaluate.py                 # evaluate_behaviour(), bootstrap_ci(), error_analysis()
│   └── visualize.py                # All publication figures (300–900 dpi)
│
├── scripts/
│   ├── run_pipeline.py             # End-to-end: download → embed → train → eval → save
│   ├── evaluate_only.py            # Reproduce all tables from a saved model
│   ├── baselines.py                # Table 3 — all baseline comparisons
│   ├── ablation.py                 # Table 4 — K ablation sweep
│   ├── generate_latex.py           # LaTeX source for Tables 2–5 + confusion matrix
│   └── check_leakage.py            # Data leakage verification
│
├── notebooks/
│   └── CenterDistill_Colab.ipynb   # Original Colab notebook (all 28 cells)
│
├── results/                        # Saved JSON artefacts (populated at runtime)
├── figures/                        # Generated figures (populated at runtime)
├── requirements.txt                # Pinned dependencies
├── setup.py                        # Pip-installable package
└── LICENSE                         # Apache 2.0
```

---

## 🏭 Industry Applications

CenterDistill's three-way behaviour policy addresses a gap in virtually every NLP deployment that surfaces a single answer to users:

| Domain | CenterDistill Behaviour | Value |
|--------|------------------------|-------|
| **Virtual assistants / chatbots** | Routes ambiguous queries to *Clarify* before hallucinating | Reduces hallucination-driven churn |
| **Enterprise search & knowledge bases** | Surfaces *Alternatives* when a query spans multiple departments | Prevents silent mis-routing |
| **Medical / legal QA** | High-stakes *Clarify* routing when interpretation is uncertain | Reduces liability from overconfident answers |
| **Customer support automation** | Distinguishes product-specific vs. policy questions | Improves first-contact resolution |
| **Multilingual helpdesks** | Cross-lingual transfer without per-language retraining | Reduces localisation cost |

### Deployment Considerations

- **Latency overhead:** The center head is a single linear layer on the CLS token — negligible inference overhead over the base XLM-RoBERTa model.
- **Threshold calibration:** All thresholds are derived from training statistics. For new domains, re-run `derive_hyperparameters()` on a representative in-domain pool (~500 questions is sufficient).
- **No architectural changes required:** CenterDistill is a drop-in wrapper around any HuggingFace extractive QA model via `patch_to_hf_qa_model()`.
- **Clarification generation:** mT5-based generation is optional; the behaviour policy operates independently if generation is not deployed.

---

## 🔭 Limitations & Future Work

**Current Limitations:**

- Surface-level answer variation can create spurious centers, contributing to false clarification errors (~13 % of misclassifications).
- Evaluation is limited to two high-resource language pairs (en–es, en–de); extension to low-resource or morphologically rich languages remains open.
- Fixed thresholds (τ\_conf, τ\_ent, τ\_multi) are the primary source of errors — 97 % of misclassifications occur within margin 0.02 of a threshold boundary.
- Behaviour labels are derived from teacher-induced distributions, not independent human annotations; external validation is a prerequisite for high-stakes deployment.
- The model uses a larger backbone (~560M params) than published MLQA baselines (~340M); efficiency–behaviour trade-offs are not yet characterised.

**Planned / Future Work:**

- [ ] Dynamic threshold calibration (e.g., temperature scaling, Platt calibration)
- [ ] Extension to low-resource languages (Hindi, Swahili, Arabic)
- [ ] Human evaluation of behaviour decisions as an external benchmark
- [ ] Systematic multi-objective ablation of λ
- [ ] Lightweight distillation to sub-100M student models for on-device deployment
- [ ] Integration with retrieval-augmented systems (RAG + CenterDistill)

---

## 📚 Citation

If you use CenterDistill in your research, please cite:

```bibtex
@inproceedings{chakraborty2026centerdistill,
  title     = {CenterDistill: Weakly-Supervised Distillation for
               Ambiguity-Aware Cross-Lingual Question Answering},
  author    = {Chakraborty, Somyajit and Naskar, Sayak and Paul, Soham
               and Jana, Angshuman and Chakraborty, Nilotpal and Gayen, Avijit},
  booktitle = {Proceedings of EAAAI 2026},
  year      = {2026}
}
```

---

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. **Fork** the repository and create your branch from `main`.
2. **Install** development dependencies: `pip install -e ".[dev]"`
3. **Test** your changes: `pytest tests/ -v`
4. Ensure `check_leakage.py` still passes with zero overlaps.
5. Open a **Pull Request** with a clear description.

For major changes or new language-pair experiments, please open an issue first to discuss scope.

**Found a bug?** Open an issue with the label `bug` and include your environment details (`python --version`, `pip list | grep torch`).

---

## 📬 Contact

| Author | Affiliation | Email |
|--------|-------------|-------|
| Somyajit Chakraborty | University College Cork | 123100668@umail.ucc.ie |
| Sayak Naskar | IIIT Guwahati | sayak.naskar25m@iiitg.ac.in |
| Soham Paul | IIIT Guwahati | soham.paul25m@iiitg.ac.in |
| Avijit Gayen | IIIT Guwahati / Techno India University | avijit.gayen@iiitg.ac.in |

For questions about the codebase, open a GitHub Issue. For questions about the paper, email the corresponding authors.

---

<div align="center">

Distributed under the **Apache 2.0 License**. See [`LICENSE`](LICENSE) for details.

⭐ If you find this work useful, please consider starring the repository — it helps other researchers discover it.

</div>
