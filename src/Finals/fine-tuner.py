import sys
import subprocess
import itertools
import csv
from tabulate import tabulate
import argparse
import warnings
warnings.filterwarnings("ignore")

def main():

    parser = argparse.ArgumentParser(description="Fine tuner for the models - NLP project")
    parser.add_argument("--path", type=str, help="Path to the file to fine_tune")
    args = parser.parse_args()
    PATH = args.path

    # Hyperparameters to test
    model_names = [
        "google/mt5-base",
        "xlm-roberta-base",
        "google/byt5-base"
    ]

    learning_rates = [3e-5, 5e-5]
    batch_sizes = [16, 32]
    epochs = [15] #To change
learning_rates = [3e-5, 5e-5]

    # Launch function
    def run_experiment(model_name, lr, batch, ep=epochs):
        cmd = [
            sys.executable,
            PATH,
            "--model_name", model_name,
            "--lr", str(lr),
            "--batch_size", str(batch),
            "--epochs", str(ep),  #Epochs to run are fixed because of early stpping implemented
        ]

        #Using return string format of the script to parse results
        print("\n------------------------------------------")
        print("\n------------------------------------------")
        print("\n------------------------------------------")
        print("RUNNING:", " ".join(cmd))
        print("\n------------------------------------------")
        print("\n------------------------------------------")
        print("\n------------------------------------------")
        output = subprocess.check_output(cmd).decode()

        if "RESULT:" not in output:
            print("No RESULT line found in output!")
            return None

        line = output.split("RESULT:")[1].strip()

        result_dict = {"model": model_name, "lr": lr, "batch": batch, "epochs": ep}

        for part in line.split("|"):
            if "=" in part:
                key, val = part.strip().split("=")
                try:
                    val = float(val)
                except:
                    pass
                result_dict[key] = val

        return result_dict


    # Run all experiments
    all_results = []

    for model_name, lr, batch, ep in itertools.product(model_names, learning_rates, batch_sizes, epochs):
        res = run_experiment(model_name, lr, batch, ep)
        if res:
            all_results.append(res)


    #Table Output
    print("                \nEXPERIMENT RESULTS")
    print(
        tabulate(
            all_results,
            headers="keys",
            tablefmt="fancy_grid",
            floatfmt=".4f",
        )
    )


    # Save CSV
    if all_results:
        if(PATH.contains("1")):
            csv_file = "src/Results/finetuning_results_1.csv"
        else:
            csv_file = "src/Results/finetuning_results_2.csv"
        keys = all_results[0].keys()

        with open(csv_file, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(all_results)

        print("\nSaved results to:", csv_file)



# Entry point
if __name__ == "__main__":
    main()
