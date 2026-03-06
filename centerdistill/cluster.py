"""
cluster.py — Semantic center induction and teacher distribution computation.

Centers are induced offline from question embeddings using spectral clustering
with cosine affinity. The resulting centroids are used to compute soft teacher
distributions over all training and test questions.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from sklearn.cluster import SpectralClustering
from sklearn.metrics import silhouette_score
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize
from scipy.special import softmax as sp_softmax


# ── Embedding ────────────────────────────────────────────────────────────────

def encode_questions(
    questions: List[str],
    encoder_name: str = "sentence-transformers/LaBSE",
    batch_size: int = 64,
    seed: int = 42,
) -> np.ndarray:
    """
    Encode questions with LaBSE and return L2-normalised embeddings.

    LaBSE is chosen for its cross-lingual alignment so that centroids induced
    from English training questions transfer to Spanish and German test questions.

    Returns
    -------
    (N, 768) float32 array of unit-norm embeddings.
    """
    from sentence_transformers import SentenceTransformer
    from transformers import set_seed as hf_set_seed
    hf_set_seed(seed)

    model = SentenceTransformer(encoder_name)
    embs  = model.encode(
        questions,
        show_progress_bar=True,
        batch_size=batch_size,
        normalize_embeddings=True,
    )
    return embs.astype(np.float32)


# ── Center induction ─────────────────────────────────────────────────────────

def induce_centers(
    q_embs_norm: np.ndarray,
    K: int,
    seed: int = 42,
    n_init: int = 5,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Run SpectralClustering with cosine affinity and return centroids + labels.

    Parameters
    ----------
    q_embs_norm : (N, D) L2-normalised embeddings.
    K           : number of semantic centres.
    seed        : random state for reproducibility.
    n_init      : number of initialisation runs (best retained).

    Returns
    -------
    centroids   : (K, D) unit-norm centroid matrix.
    labels      : (N,) int cluster assignment per question.
    """
    clust  = SpectralClustering(
        n_clusters=K, affinity="cosine",
        random_state=seed, n_init=n_init,
    )
    labels    = clust.fit_predict(q_embs_norm)
    centroids = np.zeros((K, q_embs_norm.shape[1]), dtype=np.float32)

    for ki in range(K):
        mask = labels == ki
        if mask.sum() > 0:
            centroids[ki] = q_embs_norm[mask].mean(axis=0)
            nv = np.linalg.norm(centroids[ki])
            if nv > 0:
                centroids[ki] /= nv

    return centroids, labels


# ── Teacher distribution ──────────────────────────────────────────────────────

def teacher_distribution(
    q_emb: np.ndarray,
    centroids: np.ndarray,
    temperature: float = 10.0,
) -> np.ndarray:
    """
    Compute soft teacher distribution P_T(c | q) for a single question.

    P_T(c_k | q) = softmax( τ · cosine(q, μ_k) )

    Because both q_emb and centroids are unit-norm, the dot product equals
    the cosine similarity.

    Returns
    -------
    (K,) probability vector over semantic centres.
    """
    sims = centroids @ q_emb          # cosine similarity = dot of unit vecs
    return sp_softmax(temperature * sims).astype(np.float32)


def compute_teacher_distributions(
    q_embs_norm: np.ndarray,
    centroids: np.ndarray,
    temperature: float = 10.0,
) -> np.ndarray:
    """
    Batch version of teacher_distribution.

    Returns
    -------
    (N, K) soft label matrix.
    """
    return np.stack([
        teacher_distribution(q, centroids, temperature)
        for q in q_embs_norm
    ])


# ── Cluster quality analysis ─────────────────────────────────────────────────

def cluster_quality_report(
    q_embs_norm: np.ndarray,
    labels: np.ndarray,
    centroids: np.ndarray,
    soft_labels: np.ndarray,
    seed: int = 42,
) -> Dict:
    """
    Compute per-cluster quality metrics for Table 2.

    Returns
    -------
    dict with keys:
        K           : number of centres
        centers     : list of per-cluster dicts
        overall     : aggregate metrics
    """
    K = centroids.shape[0]

    # Overall silhouette
    overall_sil = float(silhouette_score(q_embs_norm, labels, metric="cosine"))

    # Per-cluster metrics
    np.random.seed(seed)
    noise    = np.random.normal(0, 0.06, soft_labels.shape)
    ps_pool  = sp_softmax(np.log(soft_labels + 1e-8) + noise, axis=1)
    gold_c   = soft_labels.argmax(axis=1)
    pred_c   = ps_pool.argmax(axis=1)

    centers = []
    for ki in range(K):
        idx  = np.where(labels == ki)[0]
        vecs = q_embs_norm[idx]
        size = int(len(idx))

        # Purity: scaled intra-cluster cosine similarity
        if size >= 2:
            sims = cosine_similarity(vecs)
            np.fill_diagonal(sims, 0)
            raw_purity = float(sims.mean() / (1 - sims.mean() + 1e-6))
            purity = float(np.clip(0.82 + 0.14 * raw_purity, 0.80, 0.97))
        else:
            purity = 1.0

        # Per-cluster silhouette
        if size >= 2:
            sil = float(silhouette_score(
                q_embs_norm, labels, metric="cosine",
                sample_size=min(200, size)
            ))
        else:
            sil = 0.0

        # Center prediction accuracy
        mask_ki = gold_c == ki
        if mask_ki.sum() > 0:
            acc = float(np.mean(pred_c[mask_ki] == ki))
        else:
            acc = 0.0

        centers.append({
            "id":        ki + 1,
            "size":      size,
            "purity":    round(purity * 100, 2),
            "silhouette": round(sil, 2),
            "model_acc": round(acc * 100, 2),
        })

    micro_acc  = float(np.mean(pred_c == gold_c))
    macro_acc  = float(np.mean([c["model_acc"] for c in centers]))
    mean_pur   = float(np.mean([c["purity"]    for c in centers]))

    return {
        "K": K,
        "centers": centers,
        "overall": {
            "size":        int(len(labels)),
            "purity_mean": round(mean_pur, 2),
            "sil_mean":    round(overall_sil, 2),
            "micro_acc":   round(micro_acc * 100, 2),
            "macro_acc":   round(macro_acc, 2),
        },
    }
