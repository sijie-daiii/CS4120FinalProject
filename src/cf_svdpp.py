#!/usr/bin/env python3
# cf_svdpp_mid.py
# -------------------------------------------------------------
# Mid‑size SVD++ collaborative‑filtering run
#
# Pipeline:
#   1. Load rating data and randomly sample 3 % for training.
#   2. Train an SVD++ model (80 factors, 12 epochs).
#   3. Save the model checkpoint to checkpoints/svdpp_mid.pkl.
#   4. Produce top‑10 recommendations for 5 000 test users.
#   5. Evaluate results with Precision@K, Recall@K, and NDCG@K.
# -------------------------------------------------------------
from pathlib import Path
import pandas as pd, json, pickle, subprocess, sys, time
from surprise import Dataset, Reader, SVDpp

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
CKPT = ROOT / "checkpoints" / "svdpp_mid.pkl"
K    = 10
SEED = 42

# Load data and sample 3 %
train_full = pd.read_csv(DATA / "ratings_train.csv")
train_df   = train_full.sample(frac=0.03, random_state=SEED).reset_index(drop=True)

test_full  = pd.read_csv(DATA / "ratings_test.csv")
test_users = test_full["user_id"].unique()[:5000]
test_df    = test_full[test_full.user_id.isin(test_users)]

print(f"Train rows: {len(train_df):,}  |  Test users: {len(test_users):,}")

# Build the Surprise dataset
reader   = Reader(rating_scale=(1, 10))
trainset = Dataset.load_from_df(train_df[["user_id", "anime_id", "rating"]], reader) \
                  .build_full_trainset()

# Train SVD++
print("Training SVD++  (n_factors=80, n_epochs=12)…")
t0 = time.time()
algo = SVDpp(
    n_factors=80,
    n_epochs=12,
    lr_all=0.005,
    reg_all=0.02,
    random_state=SEED,
)
algo.fit(trainset)
print(f"Training done in {time.time() - t0:.1f} s")

CKPT.parent.mkdir(parents=True, exist_ok=True)
pickle.dump(algo, open(CKPT, "wb"))
print(f"Model saved to {CKPT}")

# Generate Top‑K recommendations

def recommend(uid):
    seen = set(train_df.loc[train_df.user_id == uid, "anime_id"])
    cand = set(test_df["anime_id"]) - seen
    preds = [(iid, algo.predict(uid, iid).est) for iid in cand]
    return [iid for iid, _ in sorted(preds, key=lambda x: x[1], reverse=True)[:K]]

rows = [  # flat loop, no nested iteration
    {"user_id": uid, "recommendations": json.dumps(recommend(uid))}
    for uid in test_users
]

rec_path = DATA / "svdpp_recs_mid.csv"
pd.DataFrame(rows).to_csv(rec_path, index=False)
print(f"Recommendation file saved to {rec_path}")

# Evaluate
print("\nEvaluating mid SVD++ …")
subprocess.run(
    [
        sys.executable,
        str(ROOT / "src" / "evaluate.py"),
        "--rec_file",
        str(rec_path),
        "--k",
        str(K),
    ],
    check=True,
)
