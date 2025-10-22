import argparse
import pandas as pd
from datasets import Dataset, Sequence, Value # Wichtig für die Korrektur
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
    TrainingArguments,
    Trainer,
)
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report
import torch
import numpy as np

# --- Konfiguration (unverändert) ---
LANG_CONFIG = {
    "ita": {"data_path": "data/subtask2/train/ita.csv", "model_name": "dbmdz/bert-base-italian-cased"},
    "deu": {"data_path": "data/subtask2/train/deu.csv", "model_name": "bert-base-german-cased"}
}
LABEL_COLS = ["political", "racial/ethnic", "religious", "gender/sexual", "other"]

def tokenize_function(examples, tokenizer):
    return tokenizer(examples["text"], truncation=True, max_length=512)

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = torch.sigmoid(torch.from_numpy(logits)).numpy() > 0.5
    preds = preds.astype(int)
    macro_f1 = f1_score(labels, preds, average='macro', zero_division=0)
    accuracy = accuracy_score(labels, preds)
    return {'f1_macro': macro_f1, 'accuracy': accuracy}

def main():
    parser = argparse.ArgumentParser(description="Train a monolingual multi-label classifier for SemEval Task 9 Subtask 2")
    parser.add_argument("--lang", type=str, required=True, choices=['ita', 'deu'])
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    args = parser.parse_args()

    LANG = args.lang
    MODEL_NAME = LANG_CONFIG[LANG]["model_name"]
    DATA_PATH = LANG_CONFIG[LANG]["data_path"]
    print(f"\nConfiguration:\nLANGUAGE: {LANG.upper()}\nMODEL: {MODEL_NAME}\n")
    
    df = pd.read_csv(DATA_PATH)
    if "id" in df.columns: df.drop(columns=["id"], inplace=True)
    df_train, df_temp = train_test_split(df, test_size=0.2, random_state=42)
    df_val, df_test = train_test_split(df_temp, test_size=0.5, random_state=42)
    def process_dataframe(df):
        df.dropna(subset=["text"] + LABEL_COLS, inplace=True)
        df["text"] = df["text"].astype(str)
        df["labels"] = df[LABEL_COLS].values.tolist()
        return df
    df_train, df_val, df_test = process_dataframe(df_train), process_dataframe(df_val), process_dataframe(df_test)
    train_dataset, val_dataset, test_dataset = Dataset.from_pandas(df_train), Dataset.from_pandas(df_val), Dataset.from_pandas(df_test)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    train_tokenized = train_dataset.map(lambda b: tokenize_function(b, tokenizer), batched=True, num_proc=4)
    val_tokenized = val_dataset.map(lambda b: tokenize_function(b, tokenizer), batched=True, num_proc=4)
    test_tokenized = test_dataset.map(lambda b: tokenize_function(b, tokenizer), batched=True, num_proc=4)

    # KORREKTUR: Hier wird der Datentyp der 'labels'-Spalte explizit auf Float gesetzt.
    # Dies behebt den "RuntimeError: result type Float can't be cast to... Long" Fehler.
    train_tokenized = train_tokenized.cast_column("labels", Sequence(Value("float32")))
    val_tokenized = val_tokenized.cast_column("labels", Sequence(Value("float32")))
    test_tokenized = test_tokenized.cast_column("labels", Sequence(Value("float32")))

    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=len(LABEL_COLS), problem_type="multi_label_classification")
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    training_args = TrainingArguments(
        output_dir=f"./results_{LANG}",
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",
        fp16=torch.cuda.is_available(),
        dataloader_num_workers=4,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_tokenized,
        eval_dataset=val_tokenized,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    print("\nStarting training with Hugging Face Trainer...\n")
    trainer.train()

    print("\n--- Final Evaluation on Test Set ---\n")
    results = trainer.predict(test_tokenized)
    preds = torch.sigmoid(torch.from_numpy(results.predictions)).numpy() > 0.5
    preds = preds.astype(int)
    labels = results.label_ids

    print(f"\n=== Test Results ({LANG.upper()}) ===")
    print(f"Accuracy (exact match): {accuracy_score(labels, preds):.4f} | Macro F1: {f1_score(labels, preds, average='macro', zero_division=0):.4f}")
    print(classification_report(labels, preds, target_names=LABEL_COLS, digits=4, zero_division=0))

if __name__ == "__main__":
    main()