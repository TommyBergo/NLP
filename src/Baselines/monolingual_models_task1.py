import os
import argparse
import pandas as pd
import numpy as np
from datasets import load_dataset, Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
)
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report
from torch.utils.data import DataLoader
from torch.optim import AdamW
import torch

#Constants
LANG_CONFIG = {
    "ita": {
        "data_path": "data/subtask1/train/ita.csv",
        "model_name": "dbmdz/bert-base-italian-cased"
    },
    "deu": {
        "data_path": "data/subtask1/train/deu.csv",
        "model_name": "bert-base-german-cased"
    }
}

MAX_LEN = 256

# ======== FUNCTIONS ==========
def tokenize_batch(batch, tokenizer):
    return tokenizer(
        batch["text"],
        padding="max_length",
        truncation=True,
        max_length=MAX_LEN,
    )

def safe_from_pandas(df):
    ds = Dataset.from_pandas(df)
    if "__index_level_0__" in ds.column_names:
        ds = ds.remove_columns(["__index_level_0__"])
    return ds

def evaluate_model(model, loader, device, lang_name=""):
    all_preds, all_labels = [], []
    model.eval()
    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)

            outputs = model(input_ids, attention_mask=mask)[0]
            preds = torch.argmax(outputs, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='macro') 
    print(f"\n=== Test Results ({lang_name}) ===")
    print(f"Accuracy: {acc:.4f} | Macro F1: {f1:.4f}")
    print(classification_report(all_labels, all_preds, digits=4))

# ========= MAIN ==============
def main():
    parser = argparse.ArgumentParser(description="Train a monolingual classifier for SemEval Task 9 Subtask 1")

    parser.add_argument("--lang", type=str, required=True, choices=['ita', 'deu'], help="Language to train on ('ita' or 'deu')")    
    parser.add_argument("--lr", type=float, default=2e-5, help="Learning rate")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--epochs", type=int, default=6, help="Epochs number")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay")

    args = parser.parse_args()

    LANG = args.lang
    MODEL_NAME = LANG_CONFIG[LANG]["model_name"]
    DATA_PATH = LANG_CONFIG[LANG]["data_path"]
    LR = args.lr
    BATCH_SIZE = args.batch_size
    EPOCHS = args.epochs
    WD = args.weight_decay

    print(f"\nConfiguration:")
    print(f"LANGUAGE     = {LANG.upper()}")
    print(f"MODEL_NAME   = {MODEL_NAME}")
    print(f"DATA_PATH    = {DATA_PATH}")
    print(f"LR           = {LR}")
    print(f"BATCH_SIZE   = {BATCH_SIZE}")
    print(f"EPOCHS       = {EPOCHS}")
    print(f"WEIGHT_DECAY = {WD}")
    print(f"MAX_LEN      = {MAX_LEN}")
    print("")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print(f"Reading {LANG.upper()} data from {DATA_PATH}...")
    df = pd.read_csv(DATA_PATH)
    
    if "id" in df.columns:
        df.drop(columns=["id"], inplace=True)

    df_train, df_temp = train_test_split(
        df,
        test_size=0.2,
        stratify=df["polarization"],
        random_state=42,
    )
    df_val, df_test = train_test_split(
        df_temp,
        test_size=0.5,
        stratify=df_temp["polarization"],
        random_state=42,
    )

    for d in [df_train, df_val, df_test]:
        d.rename(columns={"polarization": "label"}, inplace=True)
        d.dropna(subset=["text", "label"], inplace=True)
        d["text"] = d["text"].astype(str)

    print(f"\nTrain set size: {len(df_train)}")
    print(f"Validation set size: {len(df_val)}")
    print(f"Test set size: {len(df_test)}")

    # TOKENIZER & MODEL
    print(f"\nDownloading Tokenizer and Model: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=2,
    ).to(device)

    print("\nModel info:")
    print(model.config)

    ds_train = safe_from_pandas(df_train)
    ds_val = safe_from_pandas(df_val)
    ds_test = safe_from_pandas(df_test)

    ds_train_tok = ds_train.map(lambda b: tokenize_batch(b, tokenizer), batched=True)
    ds_val_tok = ds_val.map(lambda b: tokenize_batch(b, tokenizer), batched=True)
    ds_test_tok = ds_test.map(lambda b: tokenize_batch(b, tokenizer), batched=True)

    cols = ["input_ids", "attention_mask", "label"]
    for ds in [ds_train_tok, ds_val_tok, ds_test_tok]:
        ds.set_format(type="torch", columns=cols)

    train_loader = DataLoader(ds_train_tok, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(ds_val_tok, batch_size=BATCH_SIZE, shuffle=False)

    optimizer = AdamW(model.parameters(), lr=LR, weight_decay=WD)
    criterion = torch.nn.CrossEntropyLoss()

    print("\nStarting training...\n")
    for epoch in range(EPOCHS):
        print(f"Epoch {epoch + 1}/{EPOCHS}")
        model.train()
        for batch in train_loader:
            input_ids = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)

            optimizer.zero_grad()
            outputs = model(input_ids, attention_mask=mask)[0]
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch["input_ids"].to(device)
                mask = batch["attention_mask"].to(device)
                labels = batch["label"].to(device)
                outputs = model(input_ids, attention_mask=mask)[0]
                preds = torch.argmax(outputs, dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        acc = accuracy_score(all_labels, all_preds)
        f1 = f1_score(all_labels, all_preds, average='macro')
        print(f"→ Validation Acc={acc:.4f} | Macro F1={f1:.4f}")

    test_loader = DataLoader(ds_test_tok, batch_size=BATCH_SIZE)
    print("\nRunning evaluation on test set...")
    evaluate_model(model, test_loader, device, LANG.upper())

if __name__ == "__main__":
    main()
