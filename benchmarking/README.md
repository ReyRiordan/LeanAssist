# Benchmarking Stuff

Allows for easy benchmarking of any LLM (via OpenRouter) on any of the LeanDojo datasets. Just run run_benchmark.py

Config parameters (in run_benchmark.py):
- model_name: str, determines results file path name (using same name for diff runs will overwrite results)
- model: actual model id from OpenRouter (see https://openrouter.ai/models)
- data_cat: category of dataset to use ("random" or "novel_premises")
- data_type: type of dataset to use ("train", "val", or "test")
- num_samples: number of tactics to generate at each proof state during the search
- num_workers: number of workers to run at the same time (defaults to 4)
- example_limit: how many theorems/examples from the dataset to actually benchmark on ("val" and "test" sets have 2000 each)

### Setup

pip install -r requirements.txt  
Make sure you also have Lean, Lake, etc installed.  
Create .env file: see .env.example  
Download the dataset (https://zenodo.org/records/12740403) and put it in your workspace as leandojo_benchmark_4/ directory  

### Run

Configure run_benchmark.py according to needs.
Run: python3 -m benchmarking.run_benchmark