#!/usr/bin/env python3
# save_metrics.py – gather evaluation metrics into a single CSV

from pathlib import Path
import subprocess, sys, re, numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
RECORD = ROOT / "record"
RECORD.mkdir(exist_ok=True)

def run_evaluate(rec_file: Path):
    """
    Run evaluate.py on a recommendation file and
    return Precision, Recall, and NDCG as floats.
    """
    cmd = [
        sys.executable,
        str(ROOT / "src" / "evaluate.py"),
        "--rec_file", str(rec_file),
        "--k", "10"
    ]
    output = subprocess.check_output(cmd, text=True)
    prec   = float(re.search(r"Precision@10\s*:\s*([\d.]+)", output).group(1))
    recall = float(re.search(r"Recall@10\s*:\s*([\d.]+)",    output).group(1))
    ndcg   = float(re.search(r"NDCG@10\s*:\s*([\d.]+)",      output).group(1))
    return prec, recall, ndcg

# Top-Pop baseline
p, r, n = run_evaluate(DATA / "top_pop_recs.csv")
row_pop = dict(model="Top-Pop", prec=p, recall=r, ndcg=n, rmse=np.nan)

# SVD++ on test set
p, r, n = run_evaluate(DATA / "svdpp_recs_mid.csv")
row_svd_test = dict(model="SVD++-mid (test)", prec=p, recall=r, ndcg=n, rmse=np.nan)

# SVD++ on 1 K-user dev subset
p, r, n = run_evaluate(DATA / "svdpp_recs_mid_dev.csv")
row_svd_dev = dict(model="SVD++-mid (dev-1K)", prec=p, recall=r, ndcg=n, rmse=np.nan)

# BERT regression (RMSE only)
bert_df = pd.read_csv(DATA / "bert_item_preds_mid.csv")
rmse_val = np.sqrt(((bert_df["score"] - bert_df["pred_score"]) ** 2).mean())
row_bert = dict(model="BERT-mid", prec=np.nan, recall=np.nan, ndcg=np.nan, rmse=round(rmse_val, 3))

summary = pd.DataFrame([row_pop, row_svd_test, row_svd_dev, row_bert])
out_file = RECORD / "metrics_summary.csv"
summary.to_csv(out_file, index=False)

print(f"Metrics saved to: {out_file}")
print(summary.to_markdown(index=False))
