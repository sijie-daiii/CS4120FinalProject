#!/usr/bin/env python3
# bert_reg_mid.py  –  BERT‑base + 256 tokens + 2 epochs
# ---------------------------------------------------------------
from pathlib import Path
import pandas as pd, numpy as np
from datasets import Dataset, DatasetDict
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
)

# Paths & hyperparameters
ROOT  = Path(__file__).resolve().parents[1]
DATA  = ROOT / "data"
CKPT  = ROOT / "checkpoints" / "bert_reg_mid"

MODEL_NAME = "bert-base-uncased"
MAX_LEN    = 256
BATCH      = 8
EPOCHS     = 2
LR         = 2e-5
SEED       = 42

# Load data
raw_df = (
    pd.read_csv(DATA / "clean_items.csv", usecols=["anime_id", "synopsis", "score"])
    .assign(score=lambda df: pd.to_numeric(df.score, errors="coerce"))
    .dropna(subset=["synopsis", "score"])
    .sample(frac=0.60, random_state=SEED)
)

# Split 8 / 1 / 1
train_df = raw_df.sample(frac=0.8, random_state=SEED)
temp_df  = raw_df.drop(train_df.index)
dev_df   = temp_df.sample(frac=0.5, random_state=SEED)
test_df  = temp_df.drop(dev_df.index)

def to_ds(pdf):
    return Dataset.from_pandas(pdf.reset_index(drop=True))

ds = DatasetDict(
    {
        "train": to_ds(train_df),
        "validation": to_ds(dev_df),
        "test": to_ds(test_df),
    }
)

# Tokenizer & preprocessing
tok = AutoTokenizer.from_pretrained(MODEL_NAME)

def preprocess(b):
    enc = tok(b["synopsis"], truncation=True, max_length=MAX_LEN)
    enc["labels"] = [np.float32(x) for x in b["score"]]
    return enc

ds = ds.map(preprocess, batched=True, remove_columns=["synopsis", "anime_id"])
data_collator = DataCollatorWithPadding(tok)

# Model & Trainer
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME, num_labels=1, problem_type="regression"
)

def rmse(pred):
    p, l = pred
    p = p.squeeze()
    return {"rmse": np.sqrt(np.mean((p - l) ** 2))}

args = TrainingArguments(
    output_dir="bert_reg_mid_out",
    num_train_epochs=EPOCHS,
    learning_rate=LR,
    per_device_train_batch_size=BATCH,
    per_device_eval_batch_size=BATCH,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="rmse",
    greater_is_better=False,
    seed=SEED,
    logging_steps=50,
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=ds["train"],
    eval_dataset=ds["validation"],
    data_collator=data_collator,
    tokenizer=tok,
    compute_metrics=rmse,
)

trainer.train()
trainer.save_model(CKPT)
print(f"\nMid‑size model saved to {CKPT}")

# Evaluation & export predictions
for split in ("validation", "test"):
    res = trainer.evaluate(ds[split])
    print(f"{split.capitalize()} RMSE: {res['eval_rmse']:.4f}")

preds = trainer.predict(ds["test"]).predictions.squeeze()
(
    test_df.assign(pred_score=preds)[["anime_id", "score", "pred_score"]]
    .to_csv(DATA / "bert_item_preds_mid.csv", index=False)
)
print("Predictions saved to data/bert_item_preds_mid.csv")
