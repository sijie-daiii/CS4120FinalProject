CS 4120 Final Project – Anime Recommendation Pipeline  
Top-Pop, SVD++, BERT  

---

1. Get the data  

---

• Download: [https://www.kaggle.com/datasets/dbdmobile/myanimelist-dataset](https://www.kaggle.com/datasets/dbdmobile/myanimelist-dataset)  
• Unzip, rename the folder **data**, place it in the project root  
• Keep only: anime-dataset-2023.csv, final\_animedataset.csv, users-score-2023.csv  

---

2. Install dependencies  

---

pip install -r requirements.txt  

---

3. Build tables and splits  

---

cd CS4120FinalProject/src  
python prepare\_data.py  

---  

4. Run the pipeline  

---  

python baseline.py  
python evaluate.py --rec\_file ../data/top\_pop\_recs.csv --k 10  
python bert\_reg.py    
python cf\_svdpp.py  

---  

5. Aggregate metrics  

---  

python save\_metrics.py  
