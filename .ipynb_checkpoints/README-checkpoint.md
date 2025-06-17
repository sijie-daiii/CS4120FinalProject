CS 4120 — FINAL PROJECT
Anime Recommendation Pipeline (Top-Pop · SVD++ · BERT)
Reproducible workflow on the 2023 MyAnimeList dump
======================================================================

0. Project tree (after cloning)
   ├── CS4120FinalProject/
   │   ├── data/              ← you’ll add this
   │   ├── src/               ← all python scripts live here
   │   ├── checkpoints/       ← created automatically
   │   └── record/            ← metrics & logs
   └── …

----------------------------------------------------------------------
1. Fetch the data
----------------------------------------------------------------------
   1. Download the Kaggle set  
      https://www.kaggle.com/datasets/dbdmobile/myanimelist-dataset
   2. Unzip, rename the folder **data** and drop it into
      CS4120FinalProject/
   3. Keep **only** these three CSVs inside *data/*

        anime-dataset-2023.csv
        final_animedataset.csv
        users-score-2023.csv

----------------------------------------------------------------------
2. Install dependencies  (pip example)
----------------------------------------------------------------------
   pip install -r requirements.txt
   # -- OR use conda --
   # conda env create -f environment.yml
   # conda activate cs4120-final

----------------------------------------------------------------------
3. Build the cleaned tables & splits
----------------------------------------------------------------------
   cd CS4120FinalProject/src
   python prepare_data.py

   → data/clean_items.csv
     data/ratings_train.csv   ratings_dev.csv   ratings_test.csv

----------------------------------------------------------------------
4. Run the pipeline  (recommended order)
----------------------------------------------------------------------
   # 4-A  Top-Pop popularity baseline
   python baseline.py
     ↳ data/top_pop_recs.csv

   # 4-B  Evaluate the baseline
   python evaluate.py --rec_file ../data/top_pop_recs.csv --k 10

   # 4-C  BERT score regression
   python bert_reg.py          # or bert_reg_mid.py if that’s the filename
     ↳ checkpoints/bert_reg_mid/
     ↳ data/bert_item_preds_mid.csv

   # 4-D  SVD++ collaborative filter
   python cf_svdpp.py          # or cf_svdpp_mid.py
     ↳ checkpoints/svdpp_mid.pkl
     ↳ data/svdpp_recs_mid.csv

   # 4-E  Evaluate SVD++ (optional: evaluate BERT-based recs if you create them)
   python evaluate.py --rec_file ../data/svdpp_recs_mid.csv --k 10

----------------------------------------------------------------------
5. Aggregate all metrics
----------------------------------------------------------------------
   # inside src/
   python save_metrics.py      # or open/save_metrics.ipynb and run all

   → record/metrics_summary.csv  (console prints a neat table)

----------------------------------------------------------------------
6. Folder layout after full run
----------------------------------------------------------------------
CS4120FinalProject/
├─ data/
│   ├─ anime-dataset-2023.csv
│   ├─ final_animedataset.csv
│   ├─ users-score-2023.csv
│   ├─ clean_items.csv
│   ├─ ratings_*            …
│   └─ *recs & preds csv*
├─ checkpoints/
│   ├─ bert_reg_mid/         (transformers files)
│   └─ svdpp_mid.pkl
├─ record/
│   └─ metrics_summary.csv
└─ src/
    ├─ baseline.py
    ├─ bert_reg.py           
    ├─ cf_svdpp.py           
    ├─ evaluate.py
    ├─ prepare_data.py
    └─ save_metrics.py / .ipynb
======================================================================

