import argparse
import pandas as pd
from datasets import Dataset, Sequence, Value
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
)
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report
from torch.utils.data import DataLoader
from torch.optim import AdamW
import torch

# --- Konfiguration (unverändert) ---
LANG_CONFIG = {
    "ita": {
        "data_path": "data/subtask2/train/ita.csv",
        "model_name": "dbmdz/bert-base-italian-cased"
    },
    "deu": {
        "data_path": "data/subtask2/train/deu.csv",
        "model_name": "bert-base-german-cased"
    }
}
MAX_LEN = 256
LABEL_COLS = ["political", "racial/ethnic", "religious", "gender/sexual", "other"]

# --- Funktionen (unverändert) ---
def tokenize_batch(batch, tokenizer):
    return tokenizer(
        batch["text"], padding="max_length", truncation=True, max_length=MAX_LEN
    )

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
            mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device).float()
            logits = model(input_ids, attention_mask=mask)[0]
            preds = (torch.sigmoid(logits) > 0.5).int()
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    print(f"\n=== Test Results ({lang_name}) ===")
    print(f"Accuracy (exact match): {acc:.4f} | Macro F1: {f1:.4f}")
    print(classification_report(all_labels, all_preds, target_names=LABEL_COLS, digits=4, zero_division=0))

# --- Main-Funktion (mit minimaler Korrektur) ---
def main():
    parser = argparse.ArgumentParser(description="Train a monolingual multi-label classifier for SemEval Task 9 Subtask 2")
    parser.add_argument("--lang", type=str, required=True, choices=['ita', 'deu'], help="Language to train on ('ita' or 'deu')")
    parser.add_argument("--lr", type=float, default=2e-5, help="Learning rate")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--epochs", type=int, default=6, help="Epochs number")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay")
    args = parser.parse_args()

    LANG = args.lang
    MODEL_NAME = LANG_CONFIG[LANG]["model_name"]
    DATA_PATH = LANG_CONFIG[LANG]["data_path"]

    print(f"\nConfiguration:")
    print(f"LANGUAGE     = {LANG.upper()}")
    print(f"MODEL_NAME   = {MODEL_NAME}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    df = pd.read_csv(DATA_PATH)
    if "id" in df.columns:
        df.drop(columns=["id"], inplace=True)

    df_train, df_temp = train_test_split(df, test_size=0.2, random_state=42)
    df_val, df_test = train_test_split(df_temp, test_size=0.5, random_state=42)

    for d in [df_train, df_val, df_test]:
        d.dropna(subset=["text"] + LABEL_COLS, inplace=True)
        d["text"] = d["text"].astype(str)
        d["label"] = d[LABEL_COLS].values.tolist()
        d.drop(columns=LABEL_COLS, inplace=True)

    print(f"\nTrain set size: {len(df_train)}")
    print(f"Validation set size: {len(df_val)}")
    print(f"Test set size: {len(df_test)}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=len(LABEL_COLS),
        problem_type="multi_label_classification",
    ).to(device)
    
    ds_train, ds_val, ds_test = safe_from_pandas(df_train), safe_from_pandas(df_val), safe_from_pandas(df_test)

    ds_train_tok = ds_train.map(lambda b: tokenize_batch(b, tokenizer), batched=True)
    ds_val_tok = ds_val.map(lambda b: tokenize_batch(b, tokenizer), batched=True)
    ds_test_tok = ds_test.map(lambda b: tokenize_batch(b, tokenizer), batched=True)

    # ========================== KORREKTUR HIER ==========================
    # Cast die 'label'-Spalte explizit auf Float für jedes Dataset.
    # Dies behebt den "RuntimeError: result type Float can't be cast to... Long".
    ds_train_tok = ds_train_tok.cast_column("label", Sequence(Value("float32")))
    ds_val_tok = ds_val_tok.cast_column("label", Sequence(Value("float32")))
    ds_test_tok = ds_test_tok.cast_column("label", Sequence(Value("float32")))
    
    # Setze das Format für PyTorch NACHDEM die Datentypen korrekt sind.
    cols = ["input_ids", "attention_mask", "label"]
    ds_train_tok.set_format(type="torch", columns=cols)
    ds_val_tok.set_format(type="torch", columns=cols)
    ds_test_tok.set_format(type="torch", columns=cols)
    # ======================================================================

    train_loader = DataLoader(ds_train_tok, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(ds_val_tok, batch_size=args.batch_size)
    test_loader = DataLoader(ds_test_tok, batch_size=args.batch_size)
    
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = torch.nn.BCEWithLogitsLoss()

    print("\nStarting training...\n")
    for epoch in range(args.epochs):
        model.train()
        for batch in train_loader:
            input_ids, mask, labels = batch["input_ids"].to(device), batch["attention_mask"].to(device), batch["label"].to(device)
            optimizer.zero_grad()
            logits = model(input_ids, attention_mask=mask)[0]
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

        model.eval()
        all_preds, all_labels_val = [], []
        with torch.no_grad():
            for batch in val_loader:
                input_ids, mask, labels = batch["input_ids"].to(device), batch["attention_mask"].to(device), batch["label"].to(device)
                logits = model(input_ids, attention_mask=mask)[0]
                preds = (torch.sigmoid(logits) > 0.5).int()
                all_preds.extend(preds.cpu().numpy())
                all_labels_val.extend(labels.cpu().numpy())

        acc = accuracy_score(all_labels_val, all_preds)
        f1 = f1_score(all_labels_val, all_preds, average="macro", zero_division=0)
        print(f"Epoch {epoch + 1}/{args.epochs} → Validation Acc={acc:.4f} | Macro F1={f1:.4f}")

    print("\nRunning evaluation on test set...")
    evaluate_model(model, test_loader, device, LANG.upper())

if __name__ == "__main__":
    main()