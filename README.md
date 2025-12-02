# NLP Project
Repository for the NLP project — developed by **Tommaso Bergonzoni** and **Simon Muehlbauer**

---

## Project Structure
The project is organized into two main folders:

- **data/** — contains all datasets, divided by subtask and by train/dev splits (even if the dev set cannot be used, labels are missing).  
- **src/** — contains the code and is divided into 3 folders
  -**Baselines/** - contains the baseline models
    - `bilingual_model_task_1.py`
    - `bilingual_model_task_2.py`
    - `monolingual_models_task_1.py` *(requires `--lang=ita` or `--lang=deu` argument)*
    - `monolingual_models_task_2.py` *(requires `--lang=ita` or `--lang=deu` argument)*  
  -**Finals/** - contains the final models
    -'bilingual_task_1_final.py'
    -'bilingual_task_2_final.py'
    -'fine-tuner.py'
  -**Results** - contains the finetuning results. 
    - finetuning_results_bilingual_1.csv
    - finetuning_results_bilingual_1.csv

Each script is preconfigured with default arguments, including dataset paths and training parameters. However, we suggest you to check the parameter options and set them accordingly to your needs. 

---

## Usage
To execute a script, make sure your **working directory** is the **project root** (i.e., the folder containing both `data/` and `src/`).

Verify that command "pwd" gives output: 
"your/path/to/NLP"

```bash
python3 src/FolderName/script_name.py --parama1=value1 --param2=value2 ecc..
