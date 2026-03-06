"""
evaluate.py — Behaviour policy evaluation and error analysis for CenterDistill.

Three-way behaviour policy
--------------------------
ANSWER       : max_k P_S(c_k | q, d) > τ_conf
CLARIFY      : H(P_S) > τ_ent           (high distributional entropy)
ALTERNATIVES : |{k : P_S(c_k) > τ_multi}| ≥ 2   (multi-centre mass)
default      : CLARIFY

All thresholds are derived from training-pool statistics (see config.py)
and applied unchanged to every evaluation set.
"""

import numpy as np
from typing import Dict, List, Tuple
from collections import Counter
from scipy.special import softmax as sp_softmax


# ── Behaviour decision rules ─────────────────────────────────────────────────

def gold_behaviour(
    pt: np.ndarray,
    tau_conf:  float,
    tau_ent:   float,
    tau_multi: float,
) -> str:
    """Map a teacher distribution to a gold behaviour label."""
    if pt.max() > tau_conf:
        return "ANSWER"
    if -np.sum(pt * np.log(pt + 1e-10)) > tau_ent:
        return "CLARIFY"
    if (pt > tau_multi).sum() >= 2:
        return "ALTERNATIVES"
    return "CLARIFY"


def pred_behaviour(
    ps: np.ndarray,
    tau_conf:  float,
    tau_ent:   float,
    tau_multi: float,
) -> str:
    """Map a student (predicted) distribution to a behaviour label."""
    if ps.max() > tau_conf:
        return "ANSWER"
    if -np.sum(ps * np.log(ps + 1e-10)) > tau_ent:
        return "CLARIFY"
    if (ps > tau_multi).sum() >= 2:
        return "ALTERNATIVES"
    return "CLARIFY"


# ── Behaviour evaluation ──────────────────────────────────────────────────────

def evaluate_behaviour(
    PT: np.ndarray,
    PS: np.ndarray,
    tau_conf:  float,
    tau_ent:   float,
    tau_multi: float,
    K: int,
) -> Dict:
    """
    Compute behaviour accuracy, centre assignment accuracy, and
    per-cluster worst-cluster F1 (×10) as used in Table 3.

    Parameters
    ----------
    PT : (N, K) teacher distributions (gold reference).
    PS : (N, K) student/predicted distributions.

    Returns
    -------
    dict with behaviour_acc, centre_micro_acc, worst_cluster_f1,
         gold_dist, pred_dist, per_cluster
    """
    gold_labels   = [gold_behaviour(pt, tau_conf, tau_ent, tau_multi) for pt in PT]
    pred_labels   = [pred_behaviour(ps, tau_conf, tau_ent, tau_multi) for ps in PS]
    gold_centres  = [int(pt.argmax()) for pt in PT]
    pred_centres  = [int(ps.argmax()) for ps in PS]

    beh_acc        = float(np.mean([g == p for g, p in zip(gold_labels, pred_labels)]))
    centre_acc     = float(np.mean([g == p for g, p in zip(gold_centres, pred_centres)]))

    # Per-cluster worst-case F1 (behaviour accuracy × 10)
    per_cluster = []
    for ki in range(K):
        idx = [i for i, c in enumerate(gold_centres) if c == ki]
        if not idx:
            continue
        acc = sum(1 for i in idx if pred_labels[i] == gold_labels[i]) / len(idx)
        per_cluster.append({
            "center":   ki + 1,
            "n":        len(idx),
            "score":    round(acc * 10, 2),
        })

    worst = min(per_cluster, key=lambda x: x["score"])

    return {
        "behaviour_acc":    round(beh_acc * 100, 2),
        "centre_micro_acc": round(centre_acc * 100, 2),
        "worst_cluster_f1": worst,
        "gold_dist":        dict(Counter(gold_labels)),
        "pred_dist":        dict(Counter(pred_labels)),
        "per_cluster":      per_cluster,
        "gold_labels":      gold_labels,
        "pred_labels":      pred_labels,
        "gold_centres":     gold_centres,
        "pred_centres":     pred_centres,
    }


# ── Error analysis ────────────────────────────────────────────────────────────

def error_analysis(
    PT:           np.ndarray,
    PS:           np.ndarray,
    gold_labels:  List[str],
    pred_labels:  List[str],
    gold_centres: List[int],
    pred_centres: List[int],
    tau_conf:     float,
    tau_ent:      float,
) -> Dict:
    """
    Categorise misclassifications and compute threshold-proximity statistics.

    Error categories
    ----------------
    FALSE_CLARIFICATION : gold=ANSWER,      pred≠ANSWER   (over-estimated uncertainty)
    MISSED_AMBIGUITY    : gold≠ANSWER,      pred=ANSWER   (under-estimated uncertainty)
    WRONG_ALTERNATIVE   : gold↔CLARIFY/ALT confusion      (right detection, wrong type)

    Returns
    -------
    dict with breakdown counts, per-category diagnostics, confusion matrix.
    """
    errors = []
    for i, (g, p, gc, pc, pt, ps) in enumerate(
        zip(gold_labels, pred_labels, gold_centres, pred_centres, PT, PS)
    ):
        if g == p:
            continue
        entry = {
            "idx":        i,
            "gold_beh":   g,
            "pred_beh":   p,
            "gold_ctr":   gc,
            "pred_ctr":   pc,
            "max_pt":     float(pt.max()),
            "entropy_pt": float(-np.sum(pt * np.log(pt + 1e-10))),
            "max_ps":     float(ps.max()),
            "entropy_ps": float(-np.sum(ps * np.log(ps + 1e-10))),
        }
        if g == "ANSWER" and p != "ANSWER":
            entry["category"] = "FALSE_CLARIFICATION"
        elif g != "ANSWER" and p == "ANSWER":
            entry["category"] = "MISSED_AMBIGUITY"
        elif {g, p} == {"ALTERNATIVES", "CLARIFY"}:
            entry["category"] = "WRONG_ALTERNATIVE"
        else:
            entry["category"] = "OTHER"
        errors.append(entry)

    cat_counts = dict(Counter(e["category"] for e in errors))

    # Per-category diagnostics
    diagnostics = {}
    for cat in ["FALSE_CLARIFICATION", "MISSED_AMBIGUITY", "WRONG_ALTERNATIVE"]:
        subset = [e for e in errors if e["category"] == cat]
        if not subset:
            continue
        max_ps_vals = [e["max_ps"] for e in subset]
        ent_ps_vals = [e["entropy_ps"] for e in subset]
        diagnostics[cat] = {
            "n":            len(subset),
            "mean_max_ps":  round(float(np.mean(max_ps_vals)), 4),
            "std_max_ps":   round(float(np.std(max_ps_vals)), 4),
            "mean_ent_ps":  round(float(np.mean(ent_ps_vals)), 4),
        }

    # Threshold proximity (average margin errors are from the boundary)
    proximity = {}
    fc = [e for e in errors if e["category"] == "FALSE_CLARIFICATION"]
    ma = [e for e in errors if e["category"] == "MISSED_AMBIGUITY"]
    wa = [e for e in errors if e["category"] == "WRONG_ALTERNATIVE"]

    if fc:
        proximity["FALSE_CLARIFICATION"] = round(
            float(np.mean([tau_conf - e["max_ps"] for e in fc])), 4
        )
    if ma:
        proximity["MISSED_AMBIGUITY"] = round(
            float(np.mean([e["max_ps"] - tau_conf for e in ma])), 4
        )
    if wa:
        proximity["WRONG_ALTERNATIVE"] = round(
            float(np.mean([abs(e["entropy_ps"] - tau_ent) for e in wa])), 4
        )

    # Confusion matrix
    classes = ["ANSWER", "CLARIFY", "ALTERNATIVES"]
    conf    = {g: {p: 0 for p in classes} for g in classes}
    for g, p in zip(gold_labels, pred_labels):
        if g in conf and p in conf[g]:
            conf[g][p] += 1

    # Per-centre error breakdown
    n_total   = len(gold_labels)
    n_correct = n_total - len(errors)
    per_ctr   = {}
    for e in errors:
        k = e["gold_ctr"]
        per_ctr.setdefault(k, Counter())[e["category"]] += 1

    return {
        "n_total":        n_total,
        "n_errors":       len(errors),
        "accuracy":       round(n_correct / n_total * 100, 2),
        "breakdown":      cat_counts,
        "diagnostics":    diagnostics,
        "proximity":      proximity,
        "confusion":      conf,
        "per_centre":     {k: dict(v) for k, v in per_ctr.items()},
        "errors":         errors,
    }


# ── Bootstrap confidence intervals ───────────────────────────────────────────

def bootstrap_ci(
    gold_labels: List[str],
    pred_labels: List[str],
    n_resamples: int = 10_000,
    ci: float = 0.95,
    seed: int = 42,
) -> Tuple[float, float]:
    """
    Compute bootstrap confidence interval for behaviour accuracy.

    Returns
    -------
    (lower, upper) as percentages.
    """
    rng    = np.random.default_rng(seed)
    correct = np.array([g == p for g, p in zip(gold_labels, pred_labels)], dtype=float)
    N      = len(correct)
    accs   = [correct[rng.integers(0, N, N)].mean() for _ in range(n_resamples)]
    alpha  = (1 - ci) / 2
    return (
        round(float(np.quantile(accs, alpha)) * 100, 2),
        round(float(np.quantile(accs, 1 - alpha)) * 100, 2),
    )
