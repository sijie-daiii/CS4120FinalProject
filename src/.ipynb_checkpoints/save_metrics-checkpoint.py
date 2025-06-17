#!/usr/bin/env python3
from pathlib import Path
import subprocess, sys, re
import pandas as pd
import numpy as np

# Project paths
ROOT   = Path(__file__).resolve().parents[1]
DATA   = ROOT / "data"
RECORD = ROOT / "record"
RECORD.mkdir(exist_ok=True)

def run_evaluate(rec_file):
    """
    Call evaluate.py on a recommendation file and
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

# SVD++ (mid-sized sampling)
p, r, n = run_evaluate(DATA / "svdpp_recs_mid.csv")
row_svd = dict(model="SVD++-mid", prec=p, recall=r, ndcg=n, rmse=np.nan)

# BERT regression
bert_df = pd.read_csv(DATA / "bert_item_preds_mid.csv")
rmse = np.sqrt(((bert_df["score"] - bert_df["pred_score"]) ** 2).mean())
row_bert = dict(
    model="BERT-mid",
    prec=np.nan, recall=np.nan, ndcg=np.nan,
    rmse=round(rmse, 3)
)

# Combine and store
summary = pd.DataFrame([row_pop, row_svd, row_bert])
out_file = RECORD / "metrics_summary.csv"
summary.to_csv(out_file, index=False)

print(f"Metrics saved to: {out_file}")
print(summary.to_markdown(index=False))
