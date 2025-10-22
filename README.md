# NLP Project
Repository for the NLP project — developed by **Tommaso Bergonzoni** and **Simon Muehlbauer**

---

## Project Structure
The project is organized into two main folders:

- **data/** — contains all datasets, divided by subtask and by train/dev splits.  
- **src/** — contains four Python scripts, two per subtask:
  - `bilingual_model_task_1.py`
  - `bilingual_model_task_2.py`
  - `monolingual_models_task_1.py` *(requires `--len=ita` or `--len=deu` argument)*
  - `monolingual_models_task_2.py` *(requires `--len=ita` or `--len=deu` argument)*  

  The `src/` folder also includes a Jupyter notebook used during the early experimentation phase.  
  It is **not** part of the final project implementation.

Each script is preconfigured with default arguments, including dataset paths and training parameters.

---

## Usage
To execute a script, ensure your **working directory** is the **project root** (i.e., the folder containing both `data/` and `src/`).

'''bash
python script_name.py (--len=LANGUAGE in case of monolingual model)

