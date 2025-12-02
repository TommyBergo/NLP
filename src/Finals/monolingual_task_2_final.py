import argparse
import pandas as pd
from datasets import Dataset, Sequence, Value
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback, 
)
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report
import torch
import numpy as np

# CONSTANTS
LANG_CONFIG = {
    "ita": {"data_path": "data/subtask2/train/ita.csv", "model_name": "osiria/distilbert-base-italian-cased"},
    "deu": {"data_path": "data/subtask2/train/deu.csv", "model_name": "distilbert-base-german-cased"}
}
# These are the 5 target labels for Subtask 2
LABEL_COLS = ["political", "racial/ethnic", "religious", "gender/sexual", "other"]

# ======== FUNCTIONS ==========
def tokenize_function(examples, tokenizer):
    """Tokenizes the 'text' field in the examples."""
    return tokenizer(examples["text"], truncation=True, max_length=512)

def compute_metrics(eval_pred):
    """Computes Macro F1 and Accuracy for multi-label classification."""
    logits, labels = eval_pred
    # Apply sigmoid and threshold at 0.5 to get binary predictions
    preds = (torch.sigmoid(torch.from_numpy(logits)).numpy() > 0.5).astype(int)
    
    # Calculate macro F1 (main metric for early stopping)
    macro_f1 = f1_score(labels, preds, average='macro', zero_division=0)
    # Calculate exact match accuracy
    accuracy = accuracy_score(labels, preds)
    
    return {
        'f1_macro': macro_f1,
        'accuracy': accuracy
    }

# ========= MAIN ==============
def main():
    parser = argparse.ArgumentParser(description="Train a monolingual multi-label classifier for SemEval Task 9 Subtask 2")
    parser.add_argument("--lang", type=str, required=True, choices=['ita', 'deu'], help="Language to train on ('ita' or 'deu')")
    parser.add_argument("--lr", type=float, default=2e-5, help="Learning rate")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--epochs", type=int, default=3, help="Epochs number")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay")
    parser.add_argument("--patience", type=int, default=2, help="Patience for early stopping (number of epochs with no improvement)")
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

    print(f"\n--- Configuration ---")
    print(f"LANGUAGE:     {LANG.upper()}")
    print(f"MODEL_NAME:   {MODEL_NAME}")
    print(f"DATA_PATH:    {DATA_PATH}")
    print(f"LR:           {args.lr}")
    print(f"BATCH_SIZE:   {args.batch_size}")
    print(f"EPOCHS:       {args.epochs}")
    print(f"WEIGHT_DECAY: {args.weight_decay}")
    print(f"PATIENCE:     {args.patience}")
    print(f"---------------------\n")
    
    # --- Data Loading and Splitting ---
    df = pd.read_csv(DATA_PATH)
    if "id" in df.columns: df.drop(columns=["id"], inplace=True)

    # Split into Train (90%) and Temp (10%)
    # The 'temp' set will be used as both the Validation and Test set.
    df_train, df_temp = train_test_split(df, test_size=0.1, random_state=42)
    
    # Assign df_temp to both validation and test dataframes
    df_val = df_temp.copy()
    df_test = df_temp.copy()

    def process_dataframe(df):
        """Preprocesses DataFrame for Hugging Face Dataset conversion."""
        df.dropna(subset=["text"] + LABEL_COLS, inplace=True)
        df["text"] = df["text"].astype(str)
        # Combine label columns into a single list/array column for multi-label classification
        df["labels"] = df[LABEL_COLS].values.tolist()
        return df

    df_train, df_val, df_test = process_dataframe(df_train), process_dataframe(df_val), process_dataframe(df_test)
    
    print(f"\nTrain set size: {len(df_train)}")
    print(f"Validation/Test set size: {len(df_val)}") # Both use the same 10% set

    # Convert to Hugging Face Datasets
    train_dataset = Dataset.from_pandas(df_train)
    val_dataset = Dataset.from_pandas(df_val)
    test_dataset = Dataset.from_pandas(df_test)

    # --- Tokenization and Formatting ---
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    
    # Map tokenization function
    train_tokenized = train_dataset.map(lambda b: tokenize_function(b, tokenizer), batched=True, num_proc=4)
    val_tokenized = val_dataset.map(lambda b: tokenize_function(b, tokenizer), batched=True, num_proc=4)
    test_tokenized = test_dataset.map(lambda b: tokenize_function(b, tokenizer), batched=True, num_proc=4)

    # Cast 'labels' column to Sequence of float32 for multi-label task
    train_tokenized = train_tokenized.cast_column("labels", Sequence(Value("float32")))
    val_tokenized = val_tokenized.cast_column("labels", Sequence(Value("float32")))
    test_tokenized = test_tokenized.cast_column("labels", Sequence(Value("float32")))

    # --- Model and Trainer Setup ---
    # Configure model for multi-label classification
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, 
        num_labels=len(LABEL_COLS), 
        problem_type="multi_label_classification"
    )
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    training_args = TrainingArguments(
        output_dir=f"./results_{LANG}",
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        # Early Stopping configuration
        eval_strategy="epoch",  # Evaluate at the end of each epoch
        save_strategy="epoch",  # Save checkpoint at the end of each epoch
        load_best_model_at_end=True, # Load the best model (based on metric_for_best_model) at the end
        metric_for_best_model="f1_macro", # Metric to monitor for saving best model and early stopping
        early_stopping_patience=args.patience,
        # Other settings
        fp16=torch.cuda.is_available(),
        dataloader_num_workers=4,
        logging_steps=100, 
    )
    
    # Initialize EarlyStoppingCallback
    early_stopping_callback = EarlyStoppingCallback(
        early_stopping_patience=args.patience, 
        early_stopping_threshold=0.0
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_tokenized,
        # Use the 10% 'temp' set for validation (dev set)
        eval_dataset=val_tokenized, 
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        callbacks=[early_stopping_callback], 
    )

    print("\n--- Starting Training ---\n")
    trainer.train()

    print("\n--- Final Evaluation on Test Set ---\n")
    # The 'test_tokenized' here is identical to 'val_tokenized'
    results = trainer.predict(test_tokenized) 
    
    # Post-process predictions
    preds = (torch.sigmoid(torch.from_numpy(results.predictions)).numpy() > 0.5).astype(int)
    labels = results.label_ids

    # Calculate final metrics for output string
    final_acc = accuracy_score(labels, preds)
    final_f1 = f1_score(labels, preds, average='macro', zero_division=0)

    print(f"\n=== Final Results ({LANG.upper()}) ===")
    print(f"Accuracy (exact match): {final_acc:.4f}")
    print(f"Macro F1-Score: {final_f1:.4f}")
    print("\n" + classification_report(labels, preds, target_names=LABEL_COLS, digits=4, zero_division=0))

    # ------ RETURN STRING FOR OUTSIDE SCRIPT (FINE TUNER) ------
    result_string = (
        f"RESULT: model={MODEL_NAME} "
        f"| f1_{LANG.lower()}={final_f1:.4f} "
        f"| acc_{LANG.lower()}={final_acc:.4f} "
    )
    
    # Print the result string as the final output
    print(result_string)

if __name__ == "__main__":
    main()