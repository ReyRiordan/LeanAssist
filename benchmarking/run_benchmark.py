import os
import sys
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv

from benchmarking.evaluate import Evaluator, EvaluationConfig


def main():
    """Easy benchmarking config + running"""
    load_dotenv()
    OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

    # ----- CONFIG -----
    model_name = "Qwen3-235B-TEST"          # use diff name to not overwrite results from same model
    model = "qwen/qwen3-235b-a22b-2507"     # actual openrouter model id from https://openrouter.ai/models
    data_cat = "random"                     # options: "random", "novel_premises"
    data_type = "val"                       # options: "train", "val", "test"
    num_samples = 10                        # n tactics to generate at each proof state (breadth of search)
    num_workers = 4                         # concurrency
    example_limit = 4                       # n examples to evaluate on from dataset
    # -------------------

    dataset_path = Path(f"leandojo_benchmark_4/{data_cat}/{data_type}.json")
    output_path = Path(f"benchmarking/results/{model_name}_{data_cat}_{data_type}")
    config = EvaluationConfig(
        model = model,
        dataset_path = dataset_path,
        output_path = output_path,
        num_samples = num_samples,
        num_workers = num_workers,
    )

    evaluator = Evaluator(config, OPENROUTER_API_KEY)
    try:
        evaluator.evaluate(example_limit=example_limit)
        print("COMPLETE")
    except Exception as e:
        print(f"FAILED: {e}")


if __name__ == "__main__":
    main()
