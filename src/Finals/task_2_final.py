import os
import argparse
import pandas as pd
import numpy as np
import datasets
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
)
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report
from torch.utils.data import DataLoader
from torch.optim import AdamW
import torch

#remove warnings
import warnings
warnings.filterwarnings("ignore")

#CONSTANTS
ITA_DATA = "data/subtask2/train/ita.csv"
DEU_DATA = "data/subtask2/train/deu.csv"
SPA_DATA = "data/subtask2/train/spa.csv"
ENG_DATA = "data/subtask2/train/eng.csv"
MAX_LEN = 256
LABEL_COLS = ["political", "racial/ethnic", "religious", "gender/sexual", "other"]

# ======= FUNCTIONS =======
def load_data():
    print("Reading ita_data...")
    df_ita = pd.read_csv(ITA_DATA)
    print("Reading deu_data...")
    df_deu = pd.read_csv(DEU_DATA)
    print("Reading spa_data...")
    df_spa = pd.read_csv(SPA_DATA)
    print("Reading eng_data...")
    df_eng = pd.read_csv(ENG_DATA)
    print("All data read.\nDimensions:")
    print(f"ITA: {df_ita.shape}")
    print(f"DEU: {df_deu.shape}")
    print(f"SPA: {df_spa.shape}")
    print(f"ENG: {df_eng.shape}")
    return df_ita, df_deu, df_spa, df_eng


def safe_from_pandas(df):
    ds = Dataset.from_pandas(df)
    if "__index_level_0__" in ds.column_names:
        ds = ds.remove_columns(["__index_level_0__"])
    return ds

def tokenize_batch(batch, tokenizer):
    return tokenizer(
        batch["text"],
        padding="max_length",
        truncation=True,
        max_length=MAX_LEN,
    )

def evaluate_model(model, loader, device, lang_name=""):
    model.eval()
    all_preds, all_labels = [], []

    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device).float()

            logits = model(input_ids, attention_mask=mask)[0]
            preds = torch.sigmoid(logits)
            preds = (preds > 0.5).int()

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average="macro")

    print(f"\n=== Test Results ({lang_name}) ===")
    print(f"Accuracy: {acc:.4f} | F1: {f1:.4f}")

    print(classification_report(
        all_labels, all_preds,
        target_names=LABEL_COLS,
        digits=4
    ))

    return acc, f1


# ========= MAIN ==============
def main():
    parser = argparse.ArgumentParser(description="Train multilabel classifier for Task 2")

    parser.add_argument(
        "--model_name", type=str, default="google/mt5-small",
        help="Model: xlm-roberta-base, bert-base-multilingual-cased, google/mt5-small, google/byt5-small"
    )
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--weight_decay", type=float, default=0.01)

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
    df_ita, df_deu, df_spa, df_eng = load_data()

    #CLEAN & SPLIT
    for df in [df_ita, df_deu, df_spa, df_eng]:
        if "id" in df.columns:
            df.drop(columns=["id"], inplace=True)

    #Italian and German splits
    df_ita_train, df_ita_val = train_test_split(df_ita,test_size=0.1,)
    df_deu_train, df_deu_val = train_test_split(df_deu,test_size=0.1,)

    #Concat
    df_train = pd.concat([df_ita_train, df_deu_train, df_spa, df_eng], ignore_index=True)
    df_val = pd.concat([df_ita_val, df_deu_val], ignore_index=True)

    #removing NaN values
    df_train = df_train.dropna(subset=["text", "political", "racial/ethnic", "religious", "gender/sexual", "other"])
    df_val = df_val.dropna(subset=["text", "political", "racial/ethnic", "religious", "gender/sexual", "other"])
    df_ita_val = df_ita_val.dropna(subset=["text", "political", "racial/ethnic", "religious", "gender/sexual", "other"])
    df_deu_val = df_deu_val.dropna(subset=["text", "political", "racial/ethnic", "religious", "gender/sexual", "other"])
    
    #Converting text in string
    df_train["text"] = df_train["text"].astype(str)
    df_val["text"] = df_val["text"].astype(str)
    df_ita_val["text"] = df_ita_val["text"].astype(str)
    df_deu_val["text"] = df_deu_val["text"].astype(str)

    print(f"\nTrain set size: {len(df_train)}")
    print(f"Validation set size: {len(df_val)}")
    print(f"Italian Test set size: {len(df_ita_val)}")
    print(f"German Test set size: {len(df_deu_val)}")

    print("Dowloading Tokenizer and Model " + MODEL_NAME)

    #Tokenizer from pretrained model
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    #Model from pretrained model with classification head
    classification_model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=5,
        problem_type="multi_label_classification",  
    ).to(device)

    print("\n\nModel Info:")
    print(classification_model.config)
    print("Tokenizer vocab size:", tokenizer.vocab_size)
    print("\n")

    # Create 'label' column as list of 5 elements and drop original columns
    for df in [df_train, df_val, df_ita_val, df_deu_val]:
        df["label"] = df[LABEL_COLS].values.tolist()
        df.drop(columns=LABEL_COLS, inplace=True)
    
    #Delete unused variables
    ds_train = safe_from_pandas(df_train)
    ds_val = safe_from_pandas(df_val)
    ds_val_it = safe_from_pandas(df_ita_val)
    ds_val_de = safe_from_pandas(df_deu_val)

    #Tokenizing Datasets
    ds_train_tok = ds_train.map(lambda b: tokenize_batch(b, tokenizer), batched=True)
    ds_val_tok = ds_val.map(lambda b: tokenize_batch(b, tokenizer), batched=True)
    ds_it_tok = ds_val_it.map(lambda b: tokenize_batch(b, tokenizer), batched=True)
    ds_de_tok = ds_val_de.map(lambda b: tokenize_batch(b, tokenizer), batched=True)

    # Format for PyTorch
    for ds in [ds_train_tok, ds_val_tok, ds_it_tok, ds_de_tok]:
        ds.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])
        ds = ds.cast_column("label", datasets.Sequence(datasets.Value("float32")))

    # Cast label column to float32   
    ds_train_tok = ds_train_tok.cast_column("label", datasets.Sequence(datasets.Value("float32")))
    ds_val_tok   = ds_val_tok.cast_column("label", datasets.Sequence(datasets.Value("float32")))
    ds_it_tok = ds_it_tok.cast_column("label", datasets.Sequence(datasets.Value("float32")))
    ds_de_tok = ds_de_tok.cast_column("label", datasets.Sequence(datasets.Value("float32")))


    #Creating Dataloaders
    train_loader = DataLoader(ds_train_tok, batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(ds_val_tok,   batch_size=BATCH_SIZE, shuffle=False)

    #Optimizer and Loss
    optimizer = AdamW(classification_model.parameters(), lr=LR, weight_decay=WD)
    criterion = torch.nn.BCEWithLogitsLoss()




    #Early Stopping Variables
    best_f1 = -1
    epochs_no_improve = 0
    patience = 3  # stop after 3 epochs without improvement
    best_state = None


    #TRAINING LOOP
    print("Starting training...")

    for epoch in range(EPOCHS):
        print(f"Epoch {epoch + 1}/{EPOCHS}")
        classification_model.train()
        total_loss = 0

        for batch in train_loader:
            input_ids = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device).float()

            optimizer.zero_grad()

            logits = classification_model(input_ids, attention_mask=mask)[0]
            loss = criterion(logits, labels)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Training Loss: {total_loss / len(train_loader):.4f}")

        # ===== Validation =====
        classification_model.eval()
        all_preds, all_labels = [], []

        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch["input_ids"].to(device)
                mask = batch["attention_mask"].to(device)
                labels = batch["label"].to(device).float()

                logits = classification_model(input_ids, attention_mask=mask)[0]
                preds = torch.sigmoid(logits)
                preds = (preds > 0.5).int()

                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        acc = accuracy_score(all_labels, all_preds)
        f1 = f1_score(all_labels, all_preds, average="macro")

        print(f"→ Val Acc={acc:.4f} | F1={f1:.4f}")

        # Early Stopping
        if f1 > best_f1:
            print(f"New best model! F1 improved {best_f1:.4f} → {f1:.4f}")
            best_f1 = f1
            epochs_no_improve = 0

            best_state = {
                "model": classification_model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "epoch": epoch,
                "best_f1": best_f1
            }
        else:
            epochs_no_improve += 1
            print(f"No improvement. Patience {epochs_no_improve}/{patience}")

        if epochs_no_improve >= patience:
            print("\nEarly stopping triggered!")
            print(f"Restoring best model from epoch {best_state['epoch'] + 1}")
            classification_model.load_state_dict(best_state["model"])
            break


    classification_model.eval()

    test_loader_it = DataLoader(ds_it_tok, batch_size=BATCH_SIZE)
    test_loader_de = DataLoader(ds_de_tok, batch_size=BATCH_SIZE)

    print("Operating Italian Test......")
    acc_it, f1_it = evaluate_model(classification_model, test_loader_it, device, "Italian")

    print("\nOperating German Test......")
    acc_de, f1_de = evaluate_model(classification_model, test_loader_de, device, "German")

    print("\nFINISHED.")
    print(
        f"RESULT: model={MODEL_NAME} "
        f"| f1_it={f1_it:.4f} | f1_de={f1_de:.4f} "
        f"| acc_it={acc_it:.4f} | acc_de={acc_de:.4f}"
    )

#Launch main
if __name__ == "__main__":
    main()
