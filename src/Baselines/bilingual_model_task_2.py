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

#CONSTANTS
ITA_DATA = "data/subtask2/train/ita.csv"
DEU_DATA = "data/subtask2/train/deu.csv"
MAX_LEN = 256

# ======== FUNCTIONS ==========
def load_data():
    print("Reading ita_data...")
    df_ita = pd.read_csv(ITA_DATA)
    print("Reading deu_data...")
    df_deu = pd.read_csv(DEU_DATA)
    return df_ita, df_deu


def safe_from_pandas(df):
    ds = Dataset.from_pandas(df)
    if "__index_level_0__" in ds.column_names:
        ds = ds.remove_columns(["__index_level_0__"])
    return ds


def evaluate_model(model, loader, device, lang_name=""):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            mask      = batch["attention_mask"].to(device)
            labels    = batch["label"].to(device).float()  
            logits = model(input_ids, attention_mask=mask)[0]
            preds  = torch.sigmoid(logits)
            preds  = (preds > 0.5).int()
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    #Metrics
    acc = accuracy_score(all_labels, all_preds)
    f1  = f1_score(all_labels, all_preds, average="macro")
    print(f"\n=== Test Results ({lang_name}) ===")
    print(f"Accuracy: {acc:.4f} | F1: {f1:.4f}")
    print(classification_report(
        all_labels, all_preds,
        target_names=["political", "racial/ethnic", "religious", "gender/sexual", "other"],
        digits=4
    ))


# ========= MAIN ==============
def main():
    parser = argparse.ArgumentParser(description="Train multi-label classifier on XLM-R model")

    parser.add_argument("--model_name", type=str, default="distilbert-base-multilingual-cased", help="Name of the model: (es. xlm-roberta-base, bert-base-multilingual-cased)")
    parser.add_argument("--lr", type=float, default=2e-5, help="Learning rate")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--epochs", type=int, default=6, help="Epochs number")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay")

    args = parser.parse_args()

    #Constants
    MODEL_NAME = args.model_name
    LR = args.lr
    BATCH_SIZE = args.batch_size
    EPOCHS = args.epochs
    WD = args.weight_decay

    print(f"\nConfiguration:")
    print(f"MODEL_NAME  = {MODEL_NAME}")
    print(f"LR           = {LR}")
    print(f"BATCH_SIZE   = {BATCH_SIZE}")
    print(f"EPOCHS       = {EPOCHS}")
    print(f"WEIGHT_DECAY = {WD}")
    print(f"MAX_LEN      = {MAX_LEN}")
    print("")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    #LOAD DATA
    df_ita, df_deu = load_data()

    #CLEAN & SPLIT
    for df in [df_ita, df_deu]:
        if "id" in df.columns:
            df.drop(columns=["id"], inplace=True)

    #Italian split
    df_ita_train, df_ita_temp = train_test_split(
        df_ita,
        test_size=0.2,
    )
    df_ita_val, df_ita_test = train_test_split(
        df_ita_temp,
        test_size=0.5,
    )

    #German split
    df_deu_train, df_deu_temp = train_test_split(
        df_deu,
        test_size=0.2,
    )
    df_deu_val, df_deu_test = train_test_split(
        df_deu_temp,
        test_size=0.5,
    )

    #Concat
    df_train = pd.concat([df_ita_train, df_deu_train], ignore_index=True)
    df_val = pd.concat([df_ita_val, df_deu_val], ignore_index=True)

    #removing NaN values
    df_train = df_train.dropna(subset=["text", "political", "racial/ethnic", "religious", "gender/sexual", "other"])
    df_val = df_val.dropna(subset=["text", "political", "racial/ethnic", "religious", "gender/sexual", "other"])
    df_ita_test = df_ita_test.dropna(subset=["text", "political", "racial/ethnic", "religious", "gender/sexual", "other"])
    df_deu_test = df_deu_test.dropna(subset=["text", "political", "racial/ethnic", "religious", "gender/sexual", "other"])

    #Converting text in string
    df_train["text"] = df_train["text"].astype(str)
    df_val["text"] = df_val["text"].astype(str)
    df_ita_test["text"] = df_ita_test["text"].astype(str)
    df_deu_test["text"] = df_deu_test["text"].astype(str)

    print(f"\nTrain set size: {len(df_train)}")
    print(f"Validation set size: {len(df_val)}")
    print(f"Italian Test set size: {len(df_ita_test)}")
    print(f"German Test set size: {len(df_deu_test)}")

    print("Dowloading Tokenizer and Model " + MODEL_NAME)

    #Tokenizer from pretrained model
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    def tokenize_batch(batch):
        return tokenizer(
            batch["text"],
            padding="max_length",
            truncation=True,
            max_length=MAX_LEN,
        )

    #Model from pretrained model with classification head
    classification_model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=5,
        problem_type="multi_label_classification",  
    ).to(device)

    print("\n\nModel Informations:\n")
    print("Tokenizer vocab size:", tokenizer.vocab_size)
    print("\n")
    print(classification_model.config)

    # Create 'label' column as list of 5 elements and drop original columns
    label_cols = ["political", "racial/ethnic", "religious", "gender/sexual", "other"]
    for df in [df_train, df_val, df_ita_test, df_deu_test]:
        df["label"] = df[label_cols].values.tolist()
        df.drop(columns=label_cols, inplace=True)
    
    #Delete unused variables
    ds_train = safe_from_pandas(df_train)
    ds_val = safe_from_pandas(df_val)
    ds_test_it = safe_from_pandas(df_ita_test)
    ds_test_de = safe_from_pandas(df_deu_test)

    #Tokenizing Datasets
    ds_train_tok = ds_train.map(tokenize_batch, batched=True)
    ds_val_tok = ds_val.map(tokenize_batch, batched=True)
    ds_test_it_tok = ds_test_it.map(tokenize_batch, batched=True)
    ds_test_de_tok = ds_test_de.map(tokenize_batch, batched=True)

    # Format for PyTorch
    cols = ["input_ids", "attention_mask", "label"]
    ds_train_tok.set_format(type="torch", columns=cols)
    ds_val_tok.set_format(type="torch", columns=cols)
    ds_test_it_tok.set_format(type="torch", columns=cols)
    ds_test_de_tok.set_format(type="torch", columns=cols)

    import datasets
    for ds in [ds_train_tok, ds_val_tok, ds_test_it_tok, ds_test_de_tok]:
        ds = ds.cast_column("label", datasets.Sequence(datasets.Value("float32")))

    print("First line of ds_train_tok in pytorch format:")
    print(ds_train_tok[0])

    #Creating Dataloaders
    train_loader = DataLoader(ds_train_tok, batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(ds_val_tok,   batch_size=BATCH_SIZE, shuffle=False)

    #Optimizer
    optimizer = AdamW(classification_model.parameters(), lr=LR, weight_decay=WD)

    #Loss Function
    criterion = torch.nn.BCEWithLogitsLoss()

    #TRAINING LOOP
    print("Starting training...")

    for epoch in range(EPOCHS):
        print(f"\nEpoch {epoch+1}/{EPOCHS}")
        classification_model.train()
        running_loss = 0.0
        for batch in train_loader:
            input_ids = batch["input_ids"].to(device)
            mask      = batch["attention_mask"].to(device)
            labels    = batch["label"].to(device).float()  

            optimizer.zero_grad()
            logits = classification_model(input_ids, attention_mask=mask)[0]
            loss   = criterion(logits, labels)         
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        print(f"Training Loss: {running_loss / len(train_loader):.4f}")

        classification_model.eval()
        all_preds, all_gold = [], []
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch["input_ids"].to(device)
                mask      = batch["attention_mask"].to(device)
                labels    = batch["label"].to(device).float()

                logits = classification_model(input_ids, attention_mask=mask)[0]
                preds  = torch.sigmoid(logits)               
                preds  = (preds > 0.5).int()
                all_preds.extend(preds.cpu().numpy())
                all_gold.extend(labels.cpu().numpy())

        acc = accuracy_score(all_gold, all_preds)
        f1  = f1_score(all_gold, all_preds, average="macro") 
        print(f"→ Validation Acc={acc:.4f} | F1={f1:.4f}")

    classification_model.eval()

    test_loader_it = DataLoader(ds_test_it_tok, batch_size=BATCH_SIZE)
    test_loader_de = DataLoader(ds_test_de_tok, batch_size=BATCH_SIZE)

    print("Operating Italian Test......")
    evaluate_model(classification_model, test_loader_it, device, "Italian")

    print("\nOperating German Test......")
    evaluate_model(classification_model, test_loader_de, device, "German")


#Launch main
if __name__ == "__main__":
    main()
