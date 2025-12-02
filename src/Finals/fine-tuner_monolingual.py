import sys
import subprocess
import itertools
import csv
from tabulate import tabulate
import argparse
import warnings
import os

# Suppress all warnings
warnings.filterwarnings("ignore")

# CONSTANTS (Only needed for setting the language-specific models for filtering/baselines)
# This block is required if you want to skip running the wrong model/language combinations
LANG_CONFIG = {
    "ita": {"model_name": "osiria/distilbert-base-italian-cased"},
    "deu": {"model_name": "distilbert-base-german-cased"}
}

def main():

    parser = argparse.ArgumentParser(description="Fine tuner for the models - NLP project (Monolingual Case)")
    parser.add_argument("--path", type=str, required=True, help="Path to the monolingual fine-tuning script to execute")
    args = parser.parse_args()
    PATH = args.path

    # Hyperparameters to test
    languages = ["ita", "deu"] # The script will train on one language at a time
    
    # Models to test (based on your request for the most complete monolingual models)
    model_names = [
        # Larger/More Complete Monolingual Models
        "dbmdz/bert-base-italian-xxl-cased",  
        "TUM/GottBERT_large", # <-- CORRECTED NAME
    ]
    
    learning_rates = [1e-5, 2e-5]
    batch_sizes = [64, 128]
    epochs = [15] # Epochs to run are fixed because of early stopping implemented

    # Launch function
    def run_experiment(model_name, lang, lr, batch, ep):
        """Runs the target script with the specified hyperparameters and parses the output."""
        
        # --- Language-Specific Model Filtering (Optional but good practice) ---
        if (("italian" in model_name.lower() or "ita" in model_name.lower()) and lang == "deu") or \
           (("german" in model_name.lower() or "deu" in model_name.lower() or "gottbert" in model_name.lower()) and lang == "ita"):
            print(f"Skipping {model_name} for {lang.upper()} (Language mismatch or specialized model).")
            return None

        # --- Command Construction (CRITICAL: passing model_name) ---
        cmd = [
            sys.executable,
            PATH,
            "--lang", lang,
            "--lr", str(lr),
            "--batch_size", str(batch),
            "--epochs", str(ep),
            "--model_name", model_name, 
        ]
        
        print("\n" + "-"*80)
        print(f"RUNNING: {' '.join(cmd)}")
        print("-" * 80 + "\n")
        
        try:
            # We capture both stdout and stderr
            output = subprocess.check_output(cmd, stderr=subprocess.STDOUT).decode()
        except subprocess.CalledProcessError as e:
            print(f"ERROR running subprocess for {lang} with {model_name}: {e.output.decode()}")
            return None
        except FileNotFoundError:
            print(f"ERROR: Python interpreter not found or path '{PATH}' is incorrect.")
            return None

        if "RESULT:" not in output:
            print("No RESULT line found in output!")
            print("--- Output Snippet ---")
            print(output[-500:])
            print("----------------------")
            return None

        line = output.split("RESULT:")[1].strip()

        # Initialize result dict including the model name used
        result_dict = {
            "language": lang, 
            "model": model_name, 
            "lr": lr, 
            "batch": batch, 
            "epochs": ep
        }
        
        # Parse the custom result string format (e.g., "| f1_ita=0.8500 | acc_ita=0.8200")
        for part in line.split("|"):
            part = part.strip()
            if "=" in part:
                key, val = part.split("=")
                try:
                    val = float(val)
                except ValueError:
                    pass
                result_dict[key] = val
        
        # We rely on the model name passed via argument, NOT LANG_CONFIG.
        return result_dict


    # Run all experiments
    all_results = []

    # Iterate over all combinations
    # Note: itertools.product provides the full Cartesian product
    for lang, model_name, lr, batch, ep in itertools.product(languages, model_names, learning_rates, batch_sizes, epochs):
        res = run_experiment(model_name, lang, lr, batch, ep)
        if res:
            all_results.append(res)


    # Table Output
    print("\n" + "="*80)
    print("MONOLINGUAL EXPERIMENT RESULTS")
    print("="*80)
    
    if not all_results:
        print("No successful experiments found.")
        return

    headers = list(all_results[0].keys())
    
    print(
        tabulate(
            all_results,
            headers=headers,
            tablefmt="fancy_grid",
            floatfmt=".4f",
        )
    )

    # Save CSV
    path_lower = PATH.lower()
    # Updated CSV file naming for better specificity
    if "subtask1" in path_lower or "1" in path_lower:
        csv_file = "src/Results/finetuning_results_monolingual_subtask1.csv"
    elif "subtask2" in path_lower or "2" in path_lower:
        csv_file = "src/Results/finetuning_results_monolingual_subtask2.csv"
    else:
        csv_file = "src/Results/finetuning_results_monolingual_unknown.csv"
        
    keys = all_results[0].keys()

    try:
        os.makedirs(os.path.dirname(csv_file), exist_ok=True)
        with open(csv_file, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(all_results)

        print("\nSaved results to:", csv_file)
    except Exception as e:
        print(f"\nERROR saving CSV: {e}")


# Entry point
if __name__ == "__main__":
    main()
