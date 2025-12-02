import os
import argparse
import pandas as pd
import numpy as np
from datasets import load_dataset, Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
     AutoModelForSeq2SeqLM,
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
ITA_DATA = "data/subtask1/train/ita.csv"
DEU_DATA = "data/subtask1/train/deu.csv"
SPA_DATA = "data/subtask1/train/spa.csv"
ENG_DATA = "data/subtask1/train/eng.csv"
MAX_LEN = 256

# ======== FUNCTIONS ==========
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


def tokenize_batch(batch, tokenizer, is_seq2seq=False):
    model_inputs =  tokenizer(
        batch["text"],
        padding="max_length",
        truncation=True,
        max_length=MAX_LEN,
    )

    if is_seq2seq:

        #Transforming numbers 0 and 1 in strings "0" and "1" for seq2seq models
        label_texts = [str(l) for l in batch["label"]]

        with tokenizer.as_target_tokenizer():
            labels_enc = tokenizer(
                label_texts,
                padding="max_length",
                truncation=True,
                max_length=4,  
            )
        
        model_inputs["labels"] = labels_enc["input_ids"]

    return model_inputs



def safe_from_pandas(df):
    ds = Dataset.from_pandas(df)
    if "__index_level_0__" in ds.column_names:
        ds = ds.remove_columns(["__index_level_0__"])
    return ds


def evaluate_model(model, loader, device, lang_name="", tokenizer=None, is_seq2seq=False):
    all_preds, all_labels = [], []
    model.eval()
    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)

            if is_seq2seq:
                gen_ids = model.generate(
                    input_ids=input_ids,
                    attention_mask=mask,
                    max_new_tokens=2,
                )
                gen_texts = tokenizer.batch_decode(gen_ids, skip_special_tokens=True)
                gen_texts = [t.strip() for t in gen_texts]

                preds = []
                for t in gen_texts:
                    if t.startswith("1"):
                        preds.append(1)
                    elif t.startswith("0"):
                        preds.append(0)
                    else:
                        preds.append(0)  # fallback
                all_preds.extend(preds)
            else:
                logits = model(input_ids, attention_mask=mask)[0]
                preds = torch.argmax(logits, dim=1)
                all_preds.extend(preds.cpu().numpy())

            all_labels.extend(labels.cpu().numpy())


    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds)
    print(f"\n=== Test Results ({lang_name}) ===")
    print(f"Accuracy: {acc:.4f} | F1: {f1:.4f}")
    print(classification_report(all_labels, all_preds, digits=4))

    return acc, f1


# ========= MAIN ==============
def main():
    parser = argparse.ArgumentParser(description="Train multi-label classifier on XLM-R model")

    parser.add_argument("--model_name", type=str, default="google/mt5-small", help="Model to use: xlm-roberta-base, bert-base-multilingual-cased, "
        "google/mt5-base, or google/byt5-base.")
    parser.add_argument("--lr", type=float, default=2e-5, help="Learning rate")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--epochs", type=int, default=20, help="Epochs number")
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
    df_ita, df_deu, df_spa, df_eng = load_data()

    #CLEAN & SPLIT
    for df in [df_ita, df_deu, df_spa, df_eng]:
        if "id" in df.columns:
            df.drop(columns=["id"], inplace=True)

    #Italian split
    df_ita_train, df_ita_val = train_test_split(
        df_ita,
        test_size=0.1,
        stratify=df_ita["polarization"],
        random_state=42,
    )

    #German split
    df_deu_train, df_deu_val = train_test_split(
        df_deu,
        test_size=0.1,
        stratify=df_deu["polarization"],
        random_state=42,
    )

    df_spa_train = df_spa.copy()  #We use english and spanish enrirely for training and we are interested in the performance of the model on these language, so we don't need a validation set.
    df_eng_train = df_eng.copy()

    
    #Merge
    df_train = pd.concat([df_ita_train, df_deu_train, df_spa_train, df_eng_train], ignore_index=True)
    df_val = pd.concat([df_ita_val, df_deu_val], ignore_index=True)

    #Rename
    for df in [df_train, df_val, df_ita_val, df_deu_val]:
        df.rename(columns={"polarization": "label"}, inplace=True)
        df.dropna(subset=["text", "label"], inplace=True)
        df["text"] = df["text"].astype(str)

    print(f"\nTrain set size: {len(df_train)}")
    print(f"Validation set size: {len(df_val)}")
    print(f"Italian Validation set size: {len(df_ita_val)}")
    print(f"German Validation set size: {len(df_deu_val)}")

    #TOKENIZER & MODEL
    print(f"\nDownloading Tokenizer and Model: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    is_seq2seq = "t5" in MODEL_NAME   #This allows to either use seq2seq or sequence classification models

    if is_seq2seq:
        model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME).to(device)
    else:
        model = AutoModelForSequenceClassification.from_pretrained(
            MODEL_NAME,
            num_labels=2,
        ).to(device)

    print("\nModel info:")
    print("Tokenizer vocab size:", tokenizer.vocab_size)
    print(model.config)

    #CONVERT TO DATASETS
    ds_train = safe_from_pandas(df_train)
    ds_val = safe_from_pandas(df_val)
    ds_val_it = safe_from_pandas(df_ita_val)
    ds_val_de = safe_from_pandas(df_deu_val)

    #TOKENIZE
    ds_train_tok = ds_train.map(lambda b: tokenize_batch(b, tokenizer, is_seq2seq), batched=True)
    ds_val_tok = ds_val.map(lambda b: tokenize_batch(b, tokenizer, is_seq2seq), batched=True)
    ds_val_it_tok = ds_val_it.map(lambda b: tokenize_batch(b, tokenizer, is_seq2seq), batched=True)
    ds_val_de_tok = ds_val_de.map(lambda b: tokenize_batch(b, tokenizer, is_seq2seq), batched=True)

    # Format for PyTorch
    if is_seq2seq:
        cols = ["input_ids", "attention_mask", "labels", "label"]  # numeric label needed for metrics
    else:
        cols = ["input_ids", "attention_mask", "label"]

    for ds in [ds_train_tok, ds_val_tok, ds_val_it_tok, ds_val_de_tok]:
        ds.set_format(type="torch", columns=cols)

    print("\nFirst training example:")
    print(ds_train_tok[0])

    #DATALOADERS
    train_loader = DataLoader(ds_train_tok, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(ds_val_tok, batch_size=BATCH_SIZE, shuffle=False)

    #OPTIMIZER & LOSS
    optimizer = AdamW(model.parameters(), lr=LR, weight_decay=WD)
    
    if not is_seq2seq:
        criterion = torch.nn.CrossEntropyLoss()

    # Early stopping
    best_f1 = -1
    epochs_no_improve = 0
    early_stop_patience = 3

    best_model_state = None

    #TRAINING LOOP
    print("\nStarting training...\n")
    for epoch in range(EPOCHS):
        print(f"Epoch {epoch + 1}/{EPOCHS}")
        model.train()

        # ===== TRAINING =====
        for batch in train_loader:
            input_ids = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            optimizer.zero_grad()

            if is_seq2seq:
                #Seq2Seq models
                labels_seq = batch["labels"].to(device)
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=mask,
                    labels=labels_seq,
                )
                loss = outputs.loss
            else:
                #XLM-R / BERT-like models
                labels = batch["label"].to(device)
                logits = model(input_ids, attention_mask=mask)[0]
                loss = criterion(logits, labels)
            
            loss.backward()
            optimizer.step()



        # ===== VALIDATION =====
        model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch["input_ids"].to(device)
                mask = batch["attention_mask"].to(device)
                
                labels_gold = batch["label"].cpu().numpy()
                all_labels.extend(labels_gold)

                if is_seq2seq:
                    #Generate predictions as text labels
                    gen_ids = model.generate(
                        input_ids=input_ids,
                        attention_mask=mask,
                        max_new_tokens=2,
                    )
                    gen_texts = tokenizer.batch_decode(gen_ids, skip_special_tokens=True)
                    # mapping "0"/"1" -> 0/1
                    gen_texts = [t.strip() for t in gen_texts]
                    preds = []
                    for t in gen_texts:
                        if t.startswith("1"):
                            preds.append(1)
                        elif t.startswith("0"):
                            preds.append(0)
                        else:  #Unexpected output
                            preds.append(0)
                    all_preds.extend(preds)
                else:
                    #Generate predictions for XLM-R / BERT-like models as logits->argmax
                    labels = batch["label"].to(device)
                    logits = model(input_ids, attention_mask=mask)[0]
                    preds = torch.argmax(logits, dim=1)
                    all_preds.extend(preds.cpu().numpy())

        acc = accuracy_score(all_labels, all_preds)
        f1 = f1_score(all_labels, all_preds)
        print(f"→ Acc={acc:.4f} | F1={f1:.4f}")

        #EARLY STOPPING
        if f1 > best_f1:
            print(f"New best model found! F1 improved from {best_f1:.4f} to {f1:.4f}")
            best_f1 = f1
            epochs_no_improve = 0

            # Save best model weights
            best_model_state = {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "epoch": epoch,
                "best_f1": best_f1
            }

        else:
            epochs_no_improve += 1
            print(f"No improvement. Patience: {epochs_no_improve}/{early_stop_patience}")

        # Stop after patience is exceeded
        if epochs_no_improve >= early_stop_patience:
            print("\nEarly stopping triggered!")
            print(f"Restoring model weights from epoch {best_model_state['epoch'] + 1}")
            model.load_state_dict(best_model_state["model"])
            break


    #FINAL TESTS
    test_loader_it = DataLoader(ds_val_it_tok, batch_size=BATCH_SIZE)
    test_loader_de = DataLoader(ds_val_de_tok, batch_size=BATCH_SIZE)

    print("\nRunning evaluation on test sets...")
    acc_it, f1_it =evaluate_model(model, test_loader_it, device, "Italian", tokenizer, is_seq2seq)
    acc_de, f1_de = evaluate_model(model, test_loader_de, device, "German", tokenizer, is_seq2seq)

    # ------ RETURN STRING FOR OUTSIDE FINETUNING SCRIPT ------
    result_string = (
        f"RESULT: model={MODEL_NAME} "
        f"| f1_it={f1_it:.4f} "
        f"| f1_de={f1_de:.4f} "
        f"| acc_it={acc_it:.4f} "
        f"| acc_de={acc_de:.4f}"
    )

    print("\n" + result_string)
    return result_string

#Launch main
if __name__ == "__main__":
    main()
