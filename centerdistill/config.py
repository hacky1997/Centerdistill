"""
config.py — CenterDistill configuration and data-driven hyperparameter derivation.

Fixed architecture constants live here directly.
Data-derived parameters (K, lambda_kl, temperature, tau_*) are computed
by `derive_hyperparameters()` after embeddings are available.
"""

import os
import json
import random
import numpy as np

# ── Default output directory (override via env var or argument) ──────────────
DEFAULT_OUTPUT_DIR = os.environ.get(
    "CENTERDISTILL_OUTPUT_DIR",
    "./outputs/centerdistill_seed42"
)

# ── Fixed config (architecture + training budget only) ──────────────────────
BASE_CFG = {
    "seed":         42,
    "base_model":   "deepset/xlm-roberta-large-squad2",
    # training
    "epochs":       4,
    "batch_size":   8,       # per-device; effective batch = 32 with grad_accum=4
    "grad_accum":   4,
    "lr":           3e-5,
    "max_len":      384,
    "stride":       128,
    "warmup_ratio": 0.1,
    "weight_decay": 0.01,
    # evaluation budget
    "n_eval":       1000,
    "n_eval_es":    1000,
    "n_eval_de":    500,
    "n_cluster":    500,     # questions used for center induction
}


def lock_seed(seed: int = 42) -> None:
    """Fix all random seeds for full reproducibility."""
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ["WANDB_DISABLED"]  = "true"


def derive_hyperparameters(
    q_embs_norm: np.ndarray,
    seed: int = 42,
    k_min: int = 4,
    k_max: int = 8,
    lambda_cap: float = 0.7,
    target_max_pt: float = 0.75,
    target_answer_frac: float = 0.20,
    verbose: bool = True,
) -> dict:
    """
    Derive K, temperature, lambda_kl, and behaviour thresholds entirely from
    training-pool embeddings. No validation-label look-up is performed.

    Parameters
    ----------
    q_embs_norm : (N, D) float32 array of L2-normalised question embeddings.
    seed        : random state for SpectralClustering.
    k_min       : minimum number of semantic centres (enforced for richness).
    k_max       : upper bound of silhouette sweep.
    lambda_cap  : upper bound on KL weight to protect span-extraction quality.
    target_max_pt        : concentration penalty threshold for temperature search.
    target_answer_frac   : minimum fraction routed to ANSWER at train time.
    verbose     : print sweep progress.

    Returns
    -------
    dict with keys: K, temperature, lambda_kl,
                    tau_conf, tau_ent, tau_multi,
                    centroids (np.ndarray),
                    silhouette_by_k (dict),
                    gold_dist (dict)
    """
    from sklearn.cluster import SpectralClustering
    from sklearn.metrics import silhouette_score
    from sklearn.preprocessing import normalize
    from scipy.special import softmax as sp_softmax

    # ── STEP 1: Optimal K via silhouette (cosine) ────────────────────────────
    if verbose:
        print("\n[1/4] Silhouette sweep (K={} … {}) …".format(2, k_max))

    sil_by_k    = {}
    labels_by_k = {}
    for k in range(2, k_max + 1):
        clust = SpectralClustering(
            n_clusters=k, affinity="cosine",
            random_state=seed, n_init=5
        )
        lbl = clust.fit_predict(q_embs_norm)
        s   = silhouette_score(q_embs_norm, lbl, metric="cosine")
        sil_by_k[k]    = float(s)
        labels_by_k[k] = lbl
        if verbose:
            print(f"   K={k}: silhouette = {s:.4f}")

    best_raw = int(max(sil_by_k, key=sil_by_k.get))
    if best_raw < k_min:
        K = int(max(
            {k: v for k, v in sil_by_k.items() if k >= k_min},
            key=lambda k: sil_by_k[k]
        ))
        if verbose:
            print(f"\n  ⚠ K={best_raw} is too coarse "
                  f"(< k_min={k_min}). Selecting K={K} instead.")
    else:
        K = best_raw

    if verbose:
        print(f"\n  → K = {K}  (silhouette = {sil_by_k[K]:.4f})")

    # Compute centroids for selected K
    lbl_best  = labels_by_k[K]
    centroids = np.zeros((K, q_embs_norm.shape[1]))
    for ki in range(K):
        mask = lbl_best == ki
        if mask.sum() > 0:
            centroids[ki] = q_embs_norm[mask].mean(axis=0)
            nv = np.linalg.norm(centroids[ki])
            if nv > 0:
                centroids[ki] /= nv

    def _teacher(q_emb, cents, tau):
        return sp_softmax(tau * (cents @ q_emb))

    # ── STEP 2: Optimal temperature ─────────────────────────────────────────
    if verbose:
        print("\n[2/4] Temperature sweep …")

    TEMP_RANGE = [0.5, 1.0, 2.0, 3.0, 5.0, 8.0, 10.0]
    chance     = 1.0 / K
    temp_stats = {}

    for tau in TEMP_RANGE:
        pt_all   = np.stack([_teacher(q, centroids, tau) for q in q_embs_norm])
        max_pt   = pt_all.max(axis=1)
        std_max  = float(max_pt.std())
        mean_max = float(max_pt.mean())
        penalty  = max(0.0, mean_max - target_max_pt)
        disc     = std_max * max(0.0, mean_max - chance) * (1.0 - penalty)
        temp_stats[tau] = {
            "std_max": std_max, "mean_max": mean_max, "disc_score": disc
        }
        if verbose:
            print(f"   τ={tau:5.1f}: std={std_max:.4f}  "
                  f"mean={mean_max:.4f}  disc={disc:.6f}")

    temperature = float(max(temp_stats, key=lambda t: temp_stats[t]["disc_score"]))
    if verbose:
        print(f"\n  → temperature = {temperature}")

    soft_labels = np.stack([_teacher(q, centroids, temperature) for q in q_embs_norm])

    # ── STEP 3: Optimal lambda ───────────────────────────────────────────────
    if verbose:
        print("\n[3/4] Lambda derivation …")

    H_pt        = float(np.mean([-np.sum(p * np.log(p + 1e-10)) for p in soft_labels]))
    kl_uniform  = float(np.log(K) - H_pt)
    ideal_lam   = H_pt / (H_pt + kl_uniform + 1e-8)
    lambda_kl   = float(min(ideal_lam, lambda_cap))
    if verbose:
        print(f"   Ideal λ* ≈ {ideal_lam:.4f}  →  capped at {lambda_cap}")
        print(f"  → lambda_kl = {lambda_kl}")

    # ── STEP 4: Behaviour thresholds ─────────────────────────────────────────
    if verbose:
        print("\n[4/4] Threshold derivation …")

    max_pt_vals = soft_labels.max(axis=1)
    ent_vals    = np.array(
        [-np.sum(p * np.log(p + 1e-10)) for p in soft_labels]
    )
    second_max  = np.sort(soft_labels, axis=1)[:, -2]

    tau_conf  = float(np.percentile(max_pt_vals, 75))
    tau_ent   = float(np.median(ent_vals[max_pt_vals <= tau_conf]))
    tau_multi = float(np.percentile(second_max, 60))

    # Ensure ≥ target_answer_frac routed to ANSWER
    for pct in range(75, 20, -5):
        tc = float(np.percentile(max_pt_vals, pct))
        if np.mean(max_pt_vals > tc) >= target_answer_frac:
            tau_conf = tc
            break

    if verbose:
        print(f"   tau_conf  = {tau_conf:.4f}")
        print(f"   tau_ent   = {tau_ent:.4f}")
        print(f"   tau_multi = {tau_multi:.4f}")

    # Gold distribution on training pool
    def _gold(pt):
        if pt.max() > tau_conf:
            return "ANSWER"
        if -np.sum(pt * np.log(pt + 1e-10)) > tau_ent:
            return "CLARIFY"
        if (pt > tau_multi).sum() >= 2:
            return "ALTERNATIVES"
        return "CLARIFY"

    from collections import Counter
    gold_dist = dict(Counter([_gold(p) for p in soft_labels]))
    if verbose:
        print(f"   Train gold distribution: {gold_dist}")
        ans_frac = gold_dist.get("ANSWER", 0) / len(soft_labels)
        print(f"   ANSWER fraction: {ans_frac:.1%}")

    return {
        "K":            K,
        "temperature":  temperature,
        "lambda_kl":    lambda_kl,
        "tau_conf":     tau_conf,
        "tau_ent":      tau_ent,
        "tau_multi":    tau_multi,
        "centroids":    centroids,
        "silhouette_by_k": sil_by_k,
        "gold_dist":    gold_dist,
    }


def save_config(cfg: dict, path: str) -> None:
    serialisable = {
        k: v.tolist() if isinstance(v, np.ndarray) else v
        for k, v in cfg.items()
        if k != "centroids"
    }
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(serialisable, f, indent=2)


def load_config(path: str) -> dict:
    with open(path) as f:
        return json.load(f)
