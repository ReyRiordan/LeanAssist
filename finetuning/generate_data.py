import json
import random
import sys
from pathlib import Path
from typing import List, Tuple, Dict


def extract_pairs(entry: Dict) -> List[Dict]:
    """Extract (state_before, tactic) pairs from single entry"""
    pairs = []
    for tactic_dict in entry.get("traced_tactics", []):
        tactic = tactic_dict.get("tactic", "")
        state = tactic_dict.get("state_before", "")

        if not tactic or not state:
            continue
        if "sorry" in tactic.lower(): # filter out sorrys
            continue

        pairs.append({
            "state_before": state,
            "tactic": tactic
        })

    return pairs


def sample_examples(all_examples: List[Dict], num_examples: int, shuffle: bool, seed: int) -> List[Dict]:
    """Sample examples"""
    if shuffle:
        random.seed(seed)
        random.shuffle(all_examples)

    # num_examples too big
    if len(all_examples) < num_examples:
        print("Configured number of examples exceeds number available")
        return all_examples

    return all_examples[:num_examples]


def load_and_extract(json_path: Path, num_examples: int, shuffle: bool, seed: int) -> Tuple[List[Dict], int]:
    """Load data from json and extract all valid tactic pairs"""
    with open(json_path) as f:
        data = json.load(f)
    data = sample_examples(data, num_examples, shuffle, seed)

    all_pairs = []
    for entry in data:
        pairs = extract_pairs(entry)
        all_pairs.extend(pairs)

    # estimate tokens (~4 char per token)
    token_estimate = 0
    for pair in all_pairs:
        token_estimate += len(pair['state_before']) + len(pair['tactic'])
        token_estimate += 50 # extra prompt tokens
    token_estimate /= 4

    return all_pairs, int(token_estimate)


def format_pair(example: Dict) -> Dict:
    """Format a pair for Fireworks fine-tuning, see https://docs.fireworks.ai/fine-tuning/fine-tuning-models#ui"""
    return {
        "messages": [
            {"role": "user", "content": (
                f"Help me prove a theorem in Lean 4. Here is my current proof state:\n{example['state_before']}\n"
                "Based on this state, suggest the next tactic I should use in Lean 4 code. Only output one tactic step in lean code and nothing else."
                )},
            {"role": "assistant", "content": f"```lean\n{example['tactic']}\n```"}
        ]
    }


def write_jsonl(examples: List[Dict], output_path: Path):
    """Write examples to JSONL output file"""
    with open(output_path, 'w') as f:
        for example in examples:
            formatted = format_pair(example)
            f.write(json.dumps(formatted) + "\n")


# ---------------


def main():
    # ----- CONFIG -----
    dataset_name = "Qwen3-4B-demo"          # use diff name to not overwrite results
    data_cat = "random"                     # options: "random", "novel_premises"
    num_examples_train = 10000                 # n training examples to extract from (~118k max?)
    num_examples_val = 1000                    # n val examples to extract from (2000 max)
    shuffle = True                          # shuffle examples before sampling
    seed = 67                               # rng seed for shuffling
    # -------------------

    output_path = Path(f"finetuning/data/{data_cat}_{dataset_name}")
    output_path.mkdir(parents=True, exist_ok=True)

    # TRAIN
    train_path = Path(f"leandojo_benchmark_4/{data_cat}/train.json")
    train_pairs, estimate_train = load_and_extract(train_path, num_examples_train, shuffle, seed)
    print(f"Extracted {len(train_pairs)} pairs for training (approx {estimate_train} tokens)")

    # VAL
    val_path = Path(f"leandojo_benchmark_4/{data_cat}/val.json")
    val_pairs, estimate_val = load_and_extract(val_path, num_examples_val, shuffle, seed)
    print(f"Extracted {len(val_pairs)} pairs for validation (approx {estimate_val} tokens)")

    # Write JSONLs
    output_path_train = output_path / "train.jsonl"
    output_path_val = output_path / "val.jsonl"
    write_jsonl(train_pairs, output_path_train)
    write_jsonl(val_pairs, output_path_val)


if __name__ == "__main__":
    main()
