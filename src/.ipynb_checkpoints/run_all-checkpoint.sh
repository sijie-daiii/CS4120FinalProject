#!/usr/bin/env bash
set -e   # 出错即停止

echo "1) prepare_data.py"
python src/prepare_data.py

echo "2) baseline.py"
python src/baseline.py

echo "3) bert_reg_mid.py"
python src/bert_reg_mid.py

echo "4) cf_svdpp_mid.py"
python src/cf_svdpp_mid.py

echo "5) evaluate Top-Pop"
python src/evaluate.py --rec_file data/top_pop_recs.csv  --k 10

echo "6) evaluate SVD++"
python src/evaluate.py --rec_file data/svdpp_recs_mid.csv --k 10

echo "7) save_metrics.py"
python src/save_metrics.py

