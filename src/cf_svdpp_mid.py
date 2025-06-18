#!/usr/bin/env python3
# cf_svdpp_mid.py
# -------------------------------------------------------------
#
# Pipeline:
#   1. Load rating data and randomly sample 3 % for training.
#   2. Train an SVD++ model (80 factors, 12 epochs).
#   3. Save the model checkpoint to checkpoints/svdpp_mid.pkl.
#   4. Produce top-10 recommendations for
#        • 5 000 test users → svdpp_recs_mid.csv
#        • 1 000 sampled dev users → svdpp_recs_mid_dev.csv
 
from pathlib import Path
import pandas as pd, json, pickle, subprocess, sys, time
from surprise import Dataset, Reader, SVDpp

# Hyper-params & paths
ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
CKPT = ROOT / "checkpoints" / "svdpp_mid.pkl"
K    = 10
SEED = 42
DEV_SAMPLE_SIZE = 1000        

# Load & split data 
train_full = pd.read_csv(DATA / "ratings_train.csv")
train_df   = train_full.sample(frac=0.03, random_state=SEED).reset_index(drop=True)

test_full  = pd.read_csv(DATA / "ratings_test.csv")
test_users = test_full["user_id"].unique()[:5_000]           
test_df    = test_full[test_full.user_id.isin(test_users)]

dev_df     = pd.read_csv(DATA / "ratings_dev.csv")
dev_users  = dev_df["user_id"].unique()[:DEV_SAMPLE_SIZE]   

print(f"Train rows: {len(train_df):,} | Test users: {len(test_users):,} | Dev users (sampled): {len(dev_users):,}")

# Build Surprise dataset
reader   = Reader(rating_scale=(1, 10))
trainset = Dataset.load_from_df(train_df[["user_id", "anime_id", "rating"]], reader) \
                  .build_full_trainset()

# Train SVD++
print("Training SVD++  (n_factors=80, n_epochs=5)…")
t0 = time.time()
algo = SVDpp(
    n_factors=80,
    n_epochs=5,
    lr_all=0.005,
    reg_all=0.02,
    random_state=SEED,
)
algo.fit(trainset)
print(f"Training done in {time.time() - t0:.1f} s")

CKPT.parent.mkdir(parents=True, exist_ok=True)
pickle.dump(algo, open(CKPT, "wb"))
print(f"Model saved to {CKPT}")

# Generate recommendations
def top_k(uid, cand_set):
    """Return top-K item ids for a user within candidate set."""
    seen = set(train_df.loc[train_df.user_id == uid, "anime_id"])
    cand = cand_set - seen
    preds = [(iid, algo.predict(uid, iid).est) for iid in cand]
    return [iid for iid, _ in sorted(preds, key=lambda x: x[1], reverse=True)[:K]]

# (a) TEST users
test_cand = set(test_df["anime_id"])
test_rows = [
    {"user_id": uid, "recommendations": json.dumps(top_k(uid, test_cand))}
    for uid in test_users
]
test_rec_path = DATA / "svdpp_recs_mid.csv"
pd.DataFrame(test_rows).to_csv(test_rec_path, index=False)
print(f"Test recommendations saved to {test_rec_path}")

# (b) DEV users (sampled) — NEW
dev_cand = set(dev_df["anime_id"])
dev_rows = [
    {"user_id": uid, "recommendations": json.dumps(top_k(uid, dev_cand))}
    for uid in dev_users
]
dev_rec_path = DATA / "svdpp_recs_mid_dev.csv"
pd.DataFrame(dev_rows).to_csv(dev_rec_path, index=False)
print(f"Dev  recommendations saved to {dev_rec_path}")


# Evaluate both files
def run_eval(rec_file, label):
    print(f"\nEvaluating mid SVD++ on **{label}** …")
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "src" / "evaluate.py"),
            "--rec_file", str(rec_file),
            "--k", str(K),
        ],
        check=True,
    )

run_eval(test_rec_path, "TEST")
run_eval(dev_rec_path,  "DEV (1K sample)")
