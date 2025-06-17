#!/usr/bin/env python3
# baseline.py

from pathlib import Path
import pandas as pd
import json

K = 10                               
ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"

# Read training/testing scores
train = pd.read_csv(DATA / "ratings_train.csv")
test  = pd.read_csv(DATA / "ratings_test.csv")

# Count the most popular K episodes in the world
top_k_items = (
    train.groupby("anime_id")
         .size()
         .sort_values(ascending=False)
         .head(K)
         .index
         .tolist()
)

# Generate the same set of recommendations for all users in the test set
rec_df = (
    test[["user_id"]]
        .drop_duplicates()
        .assign(recommendations=[json.dumps(top_k_items)] * test["user_id"].nunique())
)

# Saving
out_path = DATA / "top_pop_recs.csv"
rec_df.to_csv(out_path, index=False)
print(f"Top-Pop recommendations saved to {out_path}")
