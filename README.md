CS 4120 — Final Project
Anime Recommendation Pipeline (Top-Pop · SVD++ · BERT)
Reproducible, mid-size workflow on the MyAnimeList 2023 dump

---

1. Get the data

---

1. Download from Kaggle
   [https://www.kaggle.com/datasets/dbdmobile/myanimelist-dataset](https://www.kaggle.com/datasets/dbdmobile/myanimelist-dataset)

2. Unzip and rename the folder to  data/  and move it into the
   project root:

   CS4120FinalProject/
   ├─ data/
   └─ src/ …

3. Keep only the three CSVs we use:

   anime-dataset-2023.csv
   final\_animedataset.csv
   users-score-2023.csv

---


• Pip

```
pip install -r requirements.txt
```

---

3. Build intermediate tables

---

```
cd CS4120FinalProject/src        
python prepare_data.py
```

This writes:
data/clean\_items.csv
data/ratings\_train.csv
data/ratings\_dev.csv
data/ratings\_test.csv

---

4. Run models

---

Top-Pop baseline
python baseline.py
→ data/top\_pop\_recs.csv

BERT score regression
python bert\_reg\_mid.py
→ checkpoints/bert\_reg\_mid/
→ data/bert\_item\_preds\_mid.csv

SVD++ collaborative filtering
python cf\_svdpp\_mid.py
→ checkpoints/svdpp\_mid.pkl
→ data/svdpp\_recs\_mid.csv

---

5. Evaluate recommendations

---

From src

```
python evaluate.py --rec_file ../data/top_pop_recs.csv   --k 10
python evaluate.py --rec_file ../data/svdpp_recs_mid.csv --k 10
```

---

6. Aggregate all metrics

---

```
ru save_metrics.ipynb
```

Produces:
record/metrics\_summary.csv
and prints a summary table to the console.

---

7. Folder layout after full run

---

CS4120FinalProject/
├─ data/                 raw + processed CSVs
├─ checkpoints/          model checkpoints
│   ├─ bert\_reg\_mid/
│   └─ svdpp\_mid.pkl
├─ record/               reports & logs
│   └─ metrics\_summary.csv
└─ src/                  all pipeline scripts
├─ baseline.py
├─ bert\_reg\_mid.py
├─ cf\_svdpp\_mid.py
├─ evaluate.py
├─ prepare\_data.py
└─ save\_metrics.ipynb

---