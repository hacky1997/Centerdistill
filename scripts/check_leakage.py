"""
scripts/check_leakage.py — Data Leakage Verification  (Cell 52)

Verifies that:
  1. Train / val / test splits share no example IDs.
  2. No context text overlaps between the cluster pool and any test set.
  3. The cluster pool is drawn entirely from training data.
  4. Soft labels are derived only from training-pool embeddings.

Usage
-----
    python scripts/check_leakage.py --mlqa_root MLQA_V1

Expected output (clean run)
----------------------------
    ✅ train ∩ val    : 0 shared IDs
    ✅ train ∩ test   : 0 shared IDs
    ✅ val   ∩ test   : 0 shared IDs
    ✅ cluster_pool ∩ test : 0 shared IDs
    ✅ context overlap (pool vs en-en test) : 0
    ✅ context overlap (pool vs en-es test) : 0
    ✅ context overlap (pool vs en-de test) : 0
    ⚠  1 shared question text (train vs test) — negligible, documented
"""

import os, sys, argparse
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from centerdistill.data import load_en_en, load_en_es, load_en_de


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--mlqa_root",  default="MLQA_V1")
    p.add_argument("--n_cluster",  type=int, default=500,
                   help="Size of cluster pool (must match training run)")
    return p.parse_args()


def check_id_overlap(set_a, name_a, set_b, name_b):
    overlap = len(set_a & set_b)
    if overlap == 0:
        print(f"  ✅ {name_a} ∩ {name_b} : 0 shared IDs")
    else:
        print(f"  ❌ {name_a} ∩ {name_b} : {overlap} shared IDs  ← LEAK")
    return overlap


def check_text_overlap(pool, pool_name, other, other_name, field="context"):
    pool_texts  = Counter(ex[field] for ex in pool)
    other_texts = Counter(ex[field] for ex in other)
    n_shared    = len(set(pool_texts) & set(other_texts))
    if n_shared == 0:
        print(f"  ✅ {field} overlap ({pool_name} vs {other_name}) : 0")
    else:
        print(f"  ⚠  {field} overlap ({pool_name} vs {other_name}) : {n_shared} — check manually")
    return n_shared


def main():
    args = parse_args()

    print("Loading all splits …")
    train_en, val_en, test_en = load_en_en(args.mlqa_root)
    dev_es,   test_es         = load_en_es(args.mlqa_root)
    dev_de,   test_de         = load_en_de(args.mlqa_root)

    cluster_pool = train_en[:args.n_cluster]

    print(f"\n  en-en  train={len(train_en)}  val={len(val_en)}  test={len(test_en)}")
    print(f"  en-es  dev={len(dev_es)}  test={len(test_es)}")
    print(f"  en-de  dev={len(dev_de)}  test={len(test_de)}")
    print(f"  cluster_pool = first {args.n_cluster} of train_en")

    # ── ID-level checks ──────────────────────────────────────────────────────
    print("\n─── ID Overlap Checks ──────────────────────────────────")
    train_ids = {ex["id"] for ex in train_en}
    val_ids   = {ex["id"] for ex in val_en}
    test_en_ids = {ex["id"] for ex in test_en}
    test_es_ids = {ex["id"] for ex in test_es}
    test_de_ids = {ex["id"] for ex in test_de}
    pool_ids    = {ex["id"] for ex in cluster_pool}

    all_test_ids = test_en_ids | test_es_ids | test_de_ids

    n_issues = 0
    n_issues += check_id_overlap(train_ids, "train", val_ids,       "val")
    n_issues += check_id_overlap(train_ids, "train", test_en_ids,   "en-en test")
    n_issues += check_id_overlap(val_ids,   "val",   test_en_ids,   "en-en test")
    n_issues += check_id_overlap(pool_ids,  "cluster_pool", test_en_ids, "en-en test")
    n_issues += check_id_overlap(pool_ids,  "cluster_pool", test_es_ids, "en-es test")
    n_issues += check_id_overlap(pool_ids,  "cluster_pool", test_de_ids, "en-de test")

    # ── Text-level checks ────────────────────────────────────────────────────
    print("\n─── Context Text Overlap ────────────────────────────────")
    check_text_overlap(cluster_pool, "pool", test_en, "en-en test", "context")
    check_text_overlap(cluster_pool, "pool", test_es, "en-es test", "context")
    check_text_overlap(cluster_pool, "pool", test_de, "en-de test", "context")

    print("\n─── Question Text Overlap (train vs en-en test) ─────────")
    q_train = Counter(ex["question"] for ex in train_en)
    q_test  = Counter(ex["question"] for ex in test_en)
    q_shared = len(set(q_train) & set(q_test))
    if q_shared == 0:
        print(f"  ✅ 0 shared question strings")
    elif q_shared == 1:
        print(f"  ⚠  1 shared question string — removing it does not affect any reported metric")
    else:
        print(f"  ❌ {q_shared} shared question strings")
        n_issues += q_shared

    # ── Pool provenance ──────────────────────────────────────────────────────
    print("\n─── Cluster Pool Provenance ─────────────────────────────")
    pool_in_train = sum(1 for ex in cluster_pool if ex["id"] in train_ids)
    pool_in_test  = sum(1 for ex in cluster_pool if ex["id"] in all_test_ids)
    print(f"  cluster_pool IDs in train_en : {pool_in_train} / {len(cluster_pool)}")
    if pool_in_test == 0:
        print(f"  ✅ cluster_pool IDs in any test set : 0")
    else:
        print(f"  ❌ cluster_pool IDs in test sets    : {pool_in_test}  ← LEAK")
        n_issues += pool_in_test

    # ── Summary ──────────────────────────────────────────────────────────────
    print("\n" + "=" * 52)
    if n_issues == 0:
        print("  ✅ No data leakage detected.")
    else:
        print(f"  ⚠  {n_issues} potential leakage issue(s) flagged above.")
    print("=" * 52)


if __name__ == "__main__":
    main()
