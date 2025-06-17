#!/usr/bin/env python
# coding: utf-8
"""
prepare_data.py – MyAnimeList data cleaning + splitting

Input files (all located in the `data/` directory)
  ▸ anime-dataset-2023.csv     ← contains Synopsis / Score / Genres
  ▸ final_animedataset.csv      ← columns: user_id, anime_id, my_score
  ▸ users-score-2023.csv        ← columns: user_id, anime_id, rating

Output files (written back to `data/`)
  ▸ clean_items.csv
  ▸ ratings_train.csv
  ▸ ratings_dev.csv
  ▸ ratings_test.csv
"""

from pathlib import Path
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data"

# Load the three source CSV files
print("Loading …")
items_raw   = pd.read_csv(DATA_DIR / "anime-dataset-2023.csv",  low_memory=False)
final_raw   = pd.read_csv(DATA_DIR / "final_animedataset.csv",  low_memory=False)
scores_raw  = pd.read_csv(DATA_DIR / "users-score-2023.csv",    low_memory=False)

# Process the item table (keep synopsis + score)
# First, strip column‑name whitespace and standardize case
items_raw.columns = [c.strip() for c in items_raw.columns]

# Identify the Synopsis / Score columns (case‑insensitive)
syn_col   = next(col for col in items_raw.columns if col.lower() == "synopsis")
score_col = next(col for col in items_raw.columns if col.lower() == "score")

items = (
    items_raw[["anime_id", syn_col, score_col]]
      .rename(columns={syn_col: "synopsis", score_col: "score"})
      .dropna(subset=["synopsis", "score"])
)

# Filter out synopses whose length is outside the 30‑2000 range
items = items[items.synopsis.str.len().between(30, 2000)].reset_index(drop=True)

# Merge the two rating tables → ratings_raw
ratings_final = (
    final_raw[["user_id", "anime_id", "my_score"]]
      .rename(columns={"my_score": "rating"})
)

ratings_scores = scores_raw[["user_id", "anime_id", "rating"]]

ratings_raw = (
    pd.concat([ratings_final, ratings_scores], ignore_index=True)
      .dropna()
      .query("rating > 0")
      .astype({"user_id": int, "anime_id": int, "rating": float})
)

# Keep only anime_id values that exist in the items list
ratings_raw = ratings_raw[ratings_raw.anime_id.isin(items.anime_id)]

# Split per user into train / dev / test
#    For each user randomly take 1 record for test, 1 for dev, and put the rest in train
def split_one_user(df, seed=42):
    df = df.sample(frac=1, random_state=seed)     # Shuffle
    return df.iloc[2:], df.iloc[1:2], df.iloc[:1]  # train, dev, test

train_parts, dev_parts, test_parts = [], [], []
for _, g in ratings_raw.groupby("user_id", sort=False):
    tr, dv, te = split_one_user(g)
    train_parts.append(tr)
    dev_parts.append(dv)
    test_parts.append(te)

ratings_train = pd.concat(train_parts).reset_index(drop=True)
ratings_dev   = pd.concat(dev_parts).reset_index(drop=True)
ratings_test  = pd.concat(test_parts).reset_index(drop=True)

# Save processed results
items.to_csv(DATA_DIR / "clean_items.csv",     index=False)
ratings_train.to_csv(DATA_DIR / "ratings_train.csv", index=False)
ratings_dev.to_csv(DATA_DIR / "ratings_dev.csv",     index=False)
ratings_test.to_csv(DATA_DIR / "ratings_test.csv",   index=False)

print(f"• clean_items.csv : {len(items):,}")
print(f"• ratings_train   : {len(ratings_train):,}")
print(f"• ratings_dev     : {len(ratings_dev):,}")
print(f"• ratings_test    : {len(ratings_test):,}")
