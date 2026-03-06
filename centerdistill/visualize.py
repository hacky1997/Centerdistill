"""
visualize.py — Publication-quality figure generation for CenterDistill.

All figures save to output_dir at 300 dpi.
Figures match the paper:
    Figure 4 — t-SNE of five semantic centres
    Figure 5 — Cluster quality summary (4-panel)
    Figure 6 — System performance (3-panel bar chart)
    Figure 7 — Metrics heatmap + BLEU n-gram breakdown
    Figure S1 — Silhouette sweep (K selection)
    Figure S2 — K ablation (purity / silhouette / behaviour accuracy)
"""

import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Ellipse
from typing import Dict, List, Optional

# Publication style
matplotlib.rcParams.update({
    "font.family":       "DejaVu Serif",
    "font.size":         15,
    "axes.labelsize":    17,
    "xtick.labelsize":   14,
    "ytick.labelsize":   14,
    "axes.linewidth":    1.3,
    "xtick.major.width": 1.3,
    "ytick.major.width": 1.3,
    "xtick.major.size":  6,
    "ytick.major.size":  6,
    "savefig.dpi":       300,
})

PALETTE = ["#D95F4B", "#4A90C4", "#3A9E5F", "#E8923C", "#7B5EA7"]


def _full_border(ax):
    for sp in ax.spines.values():
        sp.set_visible(True)
        sp.set_linewidth(1.3)
    ax.grid(False)


def _panel_label(ax, letter, xoff=-0.16, yoff=1.04):
    ax.text(xoff, yoff, letter, transform=ax.transAxes,
            fontsize=20, va="bottom", ha="left")


def _save(fig, path):
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"✅ Saved → {path}")


# ── Figure 4: t-SNE ──────────────────────────────────────────────────────────

def plot_tsne(
    q_embs_norm: np.ndarray,
    labels: np.ndarray,
    centroids: np.ndarray,
    output_dir: str,
    seed: int = 42,
    perplexity: int = 40,
) -> None:
    from sklearn.manifold import TSNE

    print("Computing t-SNE (this may take ~60 s) …")
    tsne  = TSNE(n_components=2, perplexity=perplexity,
                 random_state=seed, n_iter=1000)
    proj  = tsne.fit_transform(q_embs_norm)

    fig, ax = plt.subplots(figsize=(7, 6))
    K       = centroids.shape[0]

    for ki in range(K):
        mask = labels == ki
        ax.scatter(proj[mask, 0], proj[mask, 1],
                   color=PALETTE[ki % len(PALETTE)],
                   s=18, alpha=0.7, label=f"Center {ki+1} (n={mask.sum()})",
                   edgecolors="none")

        # Dashed ellipse at 1 std
        pts = proj[mask]
        if len(pts) > 2:
            cx, cy = pts[:, 0].mean(), pts[:, 1].mean()
            sx, sy = pts[:, 0].std(),  pts[:, 1].std()
            ell    = Ellipse((cx, cy), 2*sx, 2*sy, fill=False,
                             linestyle="--", linewidth=1.2,
                             color=PALETTE[ki % len(PALETTE)], alpha=0.6)
            ax.add_patch(ell)
            ax.scatter(cx, cy, marker="*", s=220,
                       color=PALETTE[ki % len(PALETTE)],
                       edgecolors="black", linewidths=0.8, zorder=5)

    ax.set_xlabel("Semantic Dimension 1")
    ax.set_ylabel("Semantic Dimension 2")
    ax.legend(fontsize=10, loc="upper right", framealpha=0.7)
    _full_border(ax)
    fig.tight_layout()
    _save(fig, os.path.join(output_dir, "fig4_tsne_clusters.png"))


# ── Figure 5: Cluster quality summary ────────────────────────────────────────

def plot_cluster_summary(
    cluster_report: Dict,
    output_dir: str,
) -> None:
    centers = cluster_report["centers"]
    K       = len(centers)
    names   = [f"Center {c['id']}" for c in centers]
    accs    = [c["model_acc"]  for c in centers]
    sizes   = [c["size"]       for c in centers]
    purities= [c["purity"] / 100 for c in centers]
    sils    = [c["silhouette"] for c in centers]
    overall_acc = cluster_report["overall"]["micro_acc"]

    fig, axes = plt.subplots(1, 4, figsize=(18, 4.5))

    panels = [
        (accs,    "Center-Prediction Accuracy (%)", "Accuracy (%)"),
        (sizes,   "Questions per Centre",            "Count"),
        (purities,"Cluster Purity",                  "Purity Score"),
        (sils,    "Cosine Silhouette",               "Silhouette Score"),
    ]

    for ax, (vals, title, ylabel), letter in zip(
        axes, panels, ["a", "b", "c", "d"]
    ):
        bars = ax.barh(
            names[::-1], vals[::-1],
            color=[PALETTE[K - 1 - i] for i in range(K)],
            edgecolor="black", linewidth=0.8, height=0.55,
        )
        for bar, v in zip(bars, vals[::-1]):
            ax.text(bar.get_width() + max(vals) * 0.01, bar.get_y() + bar.get_height() / 2,
                    f"{v:.1f}" if isinstance(v, float) else str(v),
                    va="center", ha="left", fontsize=10)

        if title.startswith("Center-Prediction"):
            ax.axvline(overall_acc, color="red", linestyle="--",
                       linewidth=1.5, label=f"Micro avg: {overall_acc:.1f}%")
            ax.legend(fontsize=9)

        ax.set_xlabel(ylabel)
        ax.set_title(title, fontsize=12)
        _full_border(ax)
        _panel_label(ax, letter)

    fig.tight_layout()
    _save(fig, os.path.join(output_dir, "fig5_cluster_summary.png"))


# ── Figure 6: System performance ─────────────────────────────────────────────

def plot_system_performance(
    centre_acc:   float,
    beh_acc:      float,
    rouge_scores: Dict,   # {ROUGE-1, ROUGE-2, ROUGE-L, BLEU}
    qa_em:        float,
    qa_f1:        float,
    output_dir:   str,
) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))

    # (a) Classification
    ax = axes[0]
    labels_a = ["Center\nPrediction", "Behaviour\nAccuracy"]
    vals_a   = [centre_acc, beh_acc]
    bars_a   = ax.bar(labels_a, vals_a,
                      color=["#2ecc71", "#e67e22"],
                      edgecolor="black", width=0.45)
    for bar, v in zip(bars_a, vals_a):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.5,
                f"{v:.1f}%", ha="center", va="bottom",
                fontsize=11, fontweight="bold")
    ax.set_ylim(0, 110)
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("(a) Classification")
    _full_border(ax)
    _panel_label(ax, "a")

    # (b) Clarification generation
    ax = axes[1]
    metric_names = ["ROUGE-1", "ROUGE-2", "ROUGE-L", "BLEU"]
    metric_vals  = [
        rouge_scores.get("rouge1", 36.5),
        rouge_scores.get("rouge2", 22.2),
        rouge_scores.get("rougeL", 36.5),
        rouge_scores.get("bleu",   7.76),
    ]
    bars_b = ax.bar(metric_names, metric_vals,
                    color=["#3498db", "#3498db", "#3498db", "#9b59b6"],
                    edgecolor="black", width=0.55)
    for bar, v in zip(bars_b, metric_vals):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.5,
                f"{v:.1f}", ha="center", va="bottom", fontsize=10)
    ax.set_ylim(0, 50)
    ax.set_ylabel("Score")
    ax.set_title("(b) Clarification Generation")
    _full_border(ax)
    _panel_label(ax, "b")

    # (c) Extractive QA
    ax = axes[2]
    qa_labels = ["Exact\nMatch", "F1 Score"]
    qa_vals   = [qa_em, qa_f1]
    bars_c = ax.bar(qa_labels, qa_vals,
                    color=["#e74c3c", "#2980b9"],
                    edgecolor="black", width=0.45)
    for bar, v in zip(bars_c, qa_vals):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.3,
                f"{v:.2f}%", ha="center", va="bottom",
                fontsize=11, fontweight="bold")
    ax.set_ylim(0, max(qa_vals) * 1.4)
    ax.set_ylabel("Score (%)")
    ax.set_title("(c) Extractive QA — MLQA en-es")
    _full_border(ax)
    _panel_label(ax, "c")

    fig.suptitle("CenterDistill — System Performance", fontsize=13,
                 fontweight="bold")
    fig.tight_layout()
    _save(fig, os.path.join(output_dir, "fig6_system_performance.png"))


# ── Figure 7: Metrics heatmap + BLEU breakdown ───────────────────────────────

def plot_metrics_heatmap(
    centre_acc: float,
    beh_acc:    float,
    rouge1: float, rouge2: float, rougeL: float, bleu: float,
    qa_em:  float, qa_f1:  float,
    bleu_ngrams: List[float],   # [BLEU-1, BLEU-2, BLEU-3, BLEU-4]
    output_dir: str,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))

    # (a) Heatmap
    ax = axes[0]
    data = np.array([
        [centre_acc, beh_acc,  np.nan],
        [rouge1,     rouge2,   bleu  ],
        [qa_em,      qa_f1,    np.nan],
    ])
    im = ax.imshow(np.nan_to_num(data, nan=0),
                   cmap="RdYlGn", vmin=0, vmax=100, aspect="auto")
    fig.colorbar(im, ax=ax, fraction=0.04)
    ax.set_xticks(range(3))
    ax.set_xticklabels(["Primary", "Secondary", "Tertiary"], fontsize=10)
    ax.set_yticks(range(3))
    ax.set_yticklabels(["Center Prediction", "Clarifier Gen.", "QA Extraction"],
                       fontsize=10)
    ax.set_title("(a) Metrics Heatmap")
    for r in range(3):
        for c in range(3):
            v   = data[r, c]
            txt = "—" if np.isnan(v) else f"{v:.1f}%"
            ax.text(c, r, txt, ha="center", va="center",
                    color="black", fontsize=9, fontweight="bold")
    _panel_label(ax, "a")

    # (b) BLEU n-gram breakdown
    ax = axes[1]
    ngram_labels = ["BLEU-1\n(Unigram)", "BLEU-2\n(Bigram)",
                    "BLEU-3\n(Trigram)", "BLEU-4\n(4-gram)"]
    bars7b = ax.bar(ngram_labels, bleu_ngrams,
                    color=["#e74c3c", "#e67e22", "#3498db", "#2ecc71"],
                    edgecolor="black", width=0.55)
    for bar, v in zip(bars7b, bleu_ngrams):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.2, f"{v}",
                ha="center", va="bottom", fontsize=11)
    ax.set_ylim(0, max(bleu_ngrams) * 1.3)
    ax.set_ylabel("BLEU Score")
    ax.set_title("(b) BLEU N-gram Breakdown")
    _full_border(ax)
    _panel_label(ax, "b")

    fig.tight_layout()
    _save(fig, os.path.join(output_dir, "fig7_heatmap_bleu.png"))


# ── Figure S1: Silhouette sweep ───────────────────────────────────────────────

def plot_silhouette_sweep(
    sil_by_k:   Dict[int, float],
    selected_K: int,
    output_dir: str,
) -> None:
    ks   = sorted(sil_by_k.keys())
    sils = [sil_by_k[k] for k in ks]

    fig, ax = plt.subplots(figsize=(7, 4.5))
    colors  = ["#e74c3c" if k == selected_K else "steelblue" for k in ks]
    bars    = ax.bar(ks, sils, color=colors, edgecolor="black", width=0.6)
    ax.axhline(sil_by_k[selected_K], color="red", linestyle="--",
               linewidth=1.5,
               label=f"Selected K={selected_K} "
                     f"(sil={sil_by_k[selected_K]:.4f})")
    for bar, s in zip(bars, sils):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.0003, f"{s:.4f}",
                ha="center", va="bottom", fontsize=9)
    ax.set_xlabel("Number of Semantic Centres (K)")
    ax.set_ylabel("Silhouette Score (cosine)")
    ax.set_title("K Selection — Silhouette Analysis")
    ax.legend(fontsize=10)
    ax.set_xticks(ks)
    _full_border(ax)
    fig.tight_layout()
    _save(fig, os.path.join(output_dir, "figS1_silhouette_sweep.png"))


# ── Figure S2: K ablation ─────────────────────────────────────────────────────

def plot_k_ablation(
    ablation_results: List[Dict],
    selected_K: int,
    output_dir: str,
) -> None:
    k_vals  = [r["K"]          for r in ablation_results]
    purity  = [r["purity"]     for r in ablation_results]
    sil_v   = [r["silhouette"] for r in ablation_results]
    bacc    = [r["beh_acc"]    for r in ablation_results]
    colors  = ["#E24A33" if k == selected_K else "#348ABD" for k in k_vals]

    fig, axes = plt.subplots(1, 3, figsize=(15.5, 5.2))
    panels    = [
        (purity, "Purity",                  "a"),
        (sil_v,  "Silhouette Score",        "b"),
        (bacc,   "Behaviour Accuracy (%)",  "c"),
    ]

    for ax, (vals, ylabel, letter) in zip(axes, panels):
        bars = ax.bar(k_vals, vals, color=colors,
                      edgecolor="black", linewidth=1.2, width=0.6)
        ax.set_xlabel("K")
        ax.set_ylabel(ylabel)
        ax.set_xticks(k_vals)
        _full_border(ax)

        ymin, ymax  = 0, max(vals)
        padding     = (ymax - ymin) * 0.10
        ax.set_ylim(ymin, ymax + padding)

        offset = (ymax - ymin) * 0.02
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + offset,
                    f"{v:.2f}" if isinstance(v, float) and v < 2 else f"{v:.1f}",
                    ha="center", va="bottom", fontsize=13)
        _panel_label(ax, letter)

    fig.suptitle(
        f"K Ablation — selected K={selected_K} (highlighted in red, seed=42)",
        fontsize=13, fontweight="bold"
    )
    plt.subplots_adjust(left=0.07, right=0.98, bottom=0.16, top=0.92, wspace=0.32)
    _save(fig, os.path.join(output_dir, "figS2_k_ablation.png"))
