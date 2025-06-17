# download the raw dataset
#   ① visit https://www.kaggle.com/datasets/dbdmobile/myanimelist-dataset
#   ② click “Download”, unzip, rename the folder to  data
#   ③ move that data/ folder into your CS4120FinalProject root directory
#
# install dependencies
pip install -r requirements.txt

# build cleaned tables and rating splits
python src/prepare_data.py           

# Top-Pop baseline 
python src/baseline.py                  

# fine-tune BERT 
python src/bert_reg_mid.py 

# train SVD++ collaborative filter
python src/cf_svdpp_mid.py           

# evaluate Top-Pop recommendations
python src/evaluate.py --rec_file data/top_pop_recs.csv   --k 10

# evaluate SVD++ recommendations
python src/evaluate.py --rec_file data/svdpp_recs_mid.csv --k 10

# gather all metrics into a single CSV
python src/save_metrics.py
