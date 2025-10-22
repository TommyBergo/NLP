# NLP Project
Repository for the NLP project — developed by **Tommaso Bergonzoni** and **Simon Muehlbauer**

---

## Project Structure
The project is organized into two main folders:

- **data/** — contains all datasets, divided by subtask and by train/dev splits.  
- **src/** — contains four Python scripts (two per subtask).  
  For each subtask, we provide:
  - one script for **monolingual models**
  - one script for the **bilingual model**

Each script is configured with default arguments, including dataset paths.

---

## Usage
To run a script, make sure your **working directory** is the **project root** (i.e., the folder that contains both `data/` and `src/`).  
Then execute:

```bash
python script_name.py
