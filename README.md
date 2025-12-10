# LeanAssist
Math495 final project w/ Man Cao

Original LeanDojo: https://github.com/lean-dojo/LeanDojo  
Original LeanCopilot: https://github.com/lean-dojo/LeanCopilot  
Dataset/benchmark (too big for git): https://zenodo.org/records/12740403  

### Setup

pip install -r requirements.txt  
Make sure you also have Lean, Lake, cmake, etc installed.  
Create .env file: see .env.example
Download the dataset (https://zenodo.org/records/12740403) and put it in your workspace as leandojo_benchmark_4/ directory

#### Benchmarking (benchmarking/)

Allows for easy benchmarking of any LLM (via OpenRouter) on any of the LeanDojo datasets. Just set config parameters (in benchmarking/run_benchmark.py) and run with:

```bash
python3 -m benchmarking.run_benchmark
```

### Fine-tuning (finetuning/)

Easy generation of fine-tuning dataset from leandojo datasets, just config parameters (in finetuning/generate_data.py) and run with:

```bash
python3 -m finetuning.generate_data
```

### Assistant (LeanCopilot/)

First, start the server:

```bash
cd /Users/reyriordan/Documents/LeanAssist/LeanCopilot/python
uvicorn server:app --port 23337
```

Then, while the server is running, you can use it in your lean file like this:

```lean
import LeanCopilot
open Lean Meta LeanCopilot

-- Define model
def assistant : ExternalGenerator := {
  name := "model_name"
  host := "localhost"
  port := 23337
}

-- Register it
#eval registerGenerator "model_name" (.external assistant)

-- Use it
set_option LeanCopilot.suggest_tactics.model "model_name" in
example (a b c : Nat) : a + b + c = a + c + b := by
  suggest_tactics
```

See LeanCopilot/LeanCopilotTests/Demo.lean for the demo. Other commands like "search_proof" may or may not work...

To add a new OpenRouter or Fireworks model, just go to LeanCopilot/python/server.py and add an entry of the following format:

```python
"model_name": UnifiedAPIRunner(
    provider="openrouter"/"fireworks,
    model="actual_model_id",
    temperature=1.0,
    num_samples=10,
    reasoning_enabled=False,
    timeout=60
)
```

Then all you have to do is define and register it in your file, restart the file, and it should be working.