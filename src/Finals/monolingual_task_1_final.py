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

# CONSTANTS
LANG_CONFIG = {
    "ita": {
        "data_path": "data/subtask1/train/ita.csv",
        "model_name": "osiria/distilbert-base-italian-cased"
    },
    "deu": {
        "data_path": "data/subtask1/train/deu.csv",
        "model_name": "distilbert-base-german-cased"
    }
}
MAX_LEN = 256

# ======== FUNCTIONS ==========
def tokenize_batch(batch, tokenizer):
    """Tokenizes a batch of text data."""
    return tokenizer(
        batch["text"],
        padding="max_length",
        truncation=True,
        max_length=MAX_LEN,
    )

def safe_from_pandas(df):
    """Creates a Hugging Face Dataset from a pandas DataFrame, removing index columns."""
    ds = Dataset.from_pandas(df)
    if "__index_level_0__" in ds.column_names:
        ds = ds.remove_columns(["__index_level_0__"])
    return ds

def evaluate_model(model, loader, device, lang_name=""):
    """Evaluates the model on a given data loader and returns Accuracy and Macro F1."""
    all_preds, all_labels = [], []
    model.eval()
    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)

            # Get logits (outputs[0] for sequence classification)
            outputs = model(input_ids, attention_mask=mask)[0] 
            preds = torch.argmax(outputs, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0) 
    print(f"\n=== Test Results ({lang_name}) ===")
    print(f"Accuracy: {acc:.4f} | Macro F1: {f1:.4f}")
    print(classification_report(all_labels, all_preds, digits=4))
    
    # Return both accuracy and F1 score
    return acc, f1 

# ========= MAIN ==============
def main():
    parser = argparse.ArgumentParser(description="Train a monolingual classifier for SemEval Task 9 Subtask 1")

    parser.add_argument("--lang", type=str, required=True, choices=['ita', 'deu'], help="Language to train on ('ita' or 'deu')")    
    parser.add_argument("--lr", type=float, default=2e-5, help="Learning rate")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--epochs", type=int, default=6, help="Epochs number")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay")
    parser.add_argument("--patience", type=int, default=1, help="Number of epochs to wait for improvement before early stopping")
    parser.add_argument(
        "--model_name", 
        type=str, 
        default=None, 
        help="Optional: Model name to use (overrides LANG_CONFIG default)"
    )
    args = parser.parse_args()

    

    LANG = args.lang
    default_model = LANG_CONFIG[LANG]["model_name"]
    MODEL_NAME = args.model_name if args.model_name else default_model
    DATA_PATH = LANG_CONFIG[LANG]["data_path"]
    LR = args.lr
    BATCH_SIZE = args.batch_size
    EPOCHS = args.epochs
    WD = args.weight_decay
    PATIENCE = args.patience 

    print(f"\nConfiguration:")
    print(f"LANGUAGE     = {LANG.upper()}")
    print(f"MODEL_NAME   = {MODEL_NAME}")
    print(f"DATA_PATH    = {DATA_PATH}")
    print(f"LR           = {LR}")
    print(f"BATCH_SIZE   = {BATCH_SIZE}")
    print(f"EPOCHS       = {EPOCHS}")
    print(f"WEIGHT_DECAY = {WD}")
    print(f"PATIENCE     = {PATIENCE}")
    print(f"MAX_LEN      = {MAX_LEN}")
    print("")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print(f"Reading {LANG.upper()} data from {DATA_PATH}...")
    df = pd.read_csv(DATA_PATH)
    
    if "id" in df.columns:
        df.drop(columns=["id"], inplace=True)

    # Split data into Train, Validation, and Test sets (80/10/10)
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

    # Preprocessing: rename column, handle missing data, ensure text is string
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
        num_labels=2, # Binary classification for polarization
    ).to(device)

    print("\nModel info:")
    print(model.config)

    # Convert pandas DataFrames to Hugging Face Datasets
    ds_train = safe_from_pandas(df_train)
    ds_val = safe_from_pandas(df_val)
    ds_test = safe_from_pandas(df_test)

    # Tokenize the datasets
    ds_train_tok = ds_train.map(lambda b: tokenize_batch(b, tokenizer), batched=True)
    ds_val_tok = ds_val.map(lambda b: tokenize_batch(b, tokenizer), batched=True)
    ds_test_tok = ds_test.map(lambda b: tokenize_batch(b, tokenizer), batched=True)

    # Set format to PyTorch tensors for DataLoader
    cols = ["input_ids", "attention_mask", "label"]
    for ds in [ds_train_tok, ds_val_tok, ds_test_tok]:
        ds.set_format(type="torch", columns=cols)

    # Create DataLoaders
    train_loader = DataLoader(ds_train_tok, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(ds_val_tok, batch_size=BATCH_SIZE, shuffle=False)

    optimizer = AdamW(model.parameters(), lr=LR, weight_decay=WD)
    criterion = torch.nn.CrossEntropyLoss()

    # Early Stopping setup
    best_val_f1 = -np.inf
    epochs_no_improve = 0
    best_model_weights = model.state_dict() # Store the initial state as the best state

    print("\nStarting training with Early Stopping...\n")
    for epoch in range(EPOCHS):
        print(f"Epoch {epoch + 1}/{EPOCHS}")
        model.train()
        
        # --- Training Loop ---
        for batch in train_loader:
            input_ids = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)

            optimizer.zero_grad()
            outputs = model(input_ids, attention_mask=mask)[0]
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
        # --- Validation Loop ---
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

        val_acc = accuracy_score(all_labels, all_preds)
        val_f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
        print(f"→ Validation Acc={val_acc:.4f} | Macro F1={val_f1:.4f}")

        # --- Early Stopping Logic ---
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            epochs_no_improve = 0
            best_model_weights = model.state_dict() # Save the current best model
            print("  (New best validation Macro F1 found. Model state saved.)")
        else:
            epochs_no_improve += 1
            print(f"  (Validation Macro F1 did not improve. Patience: {epochs_no_improve}/{PATIENCE})")
            if epochs_no_improve == PATIENCE:
                print(f"\nEarly stopping triggered after {epoch + 1} epochs.")
                break # Stop training

    # Load the best model weights found during training
    print("\nLoading best model weights for final evaluation...")
    model.load_state_dict(best_model_weights)

    # --- Test Evaluation ---
    test_loader = DataLoader(ds_test_tok, batch_size=BATCH_SIZE)
    print("Running evaluation on test set...")
    # Original call: evaluate_model(model, test_loader, device, LANG.upper())

    # Note: We use LANG.upper() as lang_name since we're using the test set of the language currently being processed
    acc_it, f1_it = evaluate_model(model, test_loader, device, LANG.upper())


    # ------ RETURN STRING FOR OUTSIDE SCRIPT ------
    result_string = (
        f"RESULT: model={MODEL_NAME} "
        f"| f1_{LANG.lower()}={f1_it:.4f} "
        f"| acc_{LANG.lower()}={acc_it:.4f} "
    )

    # We print the result string as the script's final output for the fine-tuner
    print(result_string)

if __name__ == "__main__":
    main()