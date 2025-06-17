#!/usr/bin/env python3
# evaluate.py
# Compute Precision@k / Recall@k / NDCG@k
from pathlib import Path
import pandas as pd
import numpy as np
import argparse
import ast


def precision_at_k(recs, truth, k):
    """Precision@k."""
    return len(set(recs[:k]) & truth) / k


def recall_at_k(recs, truth, k):
    """Recall@k."""
    return len(set(recs[:k]) & truth) / len(truth) if truth else 0.0


def ndcg_at_k(recs, truth, k):
    """Normalized Discounted Cumulative Gain@k."""
    dcg = sum(1 / np.log2(i + 2) for i, a in enumerate(recs[:k]) if a in truth)
    idcg = sum(1 / np.log2(i + 2) for i in range(min(len(truth), k)))
    return dcg / idcg if idcg else 0.0


def main(rec_file: Path, k: int):
    ROOT = Path(__file__).resolve().parents[1]
    DATA = ROOT / "data"
    test = pd.read_csv(DATA / "ratings_test.csv")

    # Ground-truth: set of relevant items per user
    gt = test.groupby("user_id")["anime_id"].apply(set)

    # Load and deserialize the recommendation lists
    recs = (
        pd.read_csv(rec_file)
          .set_index("user_id")["recommendations"]
          .apply(lambda s: list(map(int, ast.literal_eval(s))))
    )

    P, R, N = [], [], []
    for uid, truth_items in gt.items():
        if uid not in recs:
            continue
        rec_list = recs[uid]
        P.append(precision_at_k(rec_list, truth_items, k))
        R.append(recall_at_k(rec_list, truth_items, k))
        N.append(ndcg_at_k(rec_list, truth_items, k))

    print(f"\nResults (k={k})")
    print("-" * 25)
    print(f"Precision@{k} : {np.mean(P):.4f}")
    print(f"Recall@{k}    : {np.mean(R):.4f}")
    print(f"NDCG@{k}      : {np.mean(N):.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--rec_file", required=True,
                        help="Path to CSV with user_id & recommendations")
    parser.add_argument("--k", type=int, default=10, help="cut-off k")
    args = parser.parse_args()
    main(Path(args.rec_file), args.k)
