import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Dict, Optional
from lean_dojo import Dojo, Theorem, LeanGitRepo, DojoInitError, DojoCrashError

from benchmarking.api_clients import OpenRouterClient, FireworksClient
from benchmarking.proof_search import ProofSearch, ProofSearchResult


@dataclass
class EvaluationConfig:
    provider: str # openrouter, fireworks
    api_key: str
    model: str
    num_samples: int
    dataset_path: str
    output_path: str
    num_workers: int = 4


class Evaluator:
    """Framework for benchmarking LLMs on LeanDojo datasets"""

    def __init__(self, config: EvaluationConfig):
        self.config = config

        # Set up API
        if config.provider == "openrouter":
            self.api_client = OpenRouterClient(config.model, config.api_key, config.num_samples)
        elif config.provider == "fireworks":
            self.api_client = FireworksClient(config.model, config.api_key, config.num_samples)
        else:
            print("UNKNOWN API PROVIDER")
        
        # Set up output stuff
        self.output_path = Path(config.output_path)
        self.output_path.mkdir(parents=True, exist_ok=True)
        self.results_file = self.output_path / "results.jsonl"
        self.summary_file = self.output_path / "summary.json"

    def load_dataset(self, limit: Optional[int] = None) -> List[Dict[str, str]]:
        """Load theorems from JSON dataset"""
        with open(self.config.dataset_path) as f:
            data = json.load(f)

        # Extract relevant fields only
        examples = []
        for item in data:
            examples.append({
                "url": item['url'], 
                "commit": item['commit'], 
                "file_path": item['file_path'], 
                "full_name": item['full_name']
            })
        if limit: examples = examples[:limit] # theorem limit

        print(f"Loaded {len(examples)} theorems from {self.config.dataset_path}")
        return examples

    def save_result(self, result: ProofSearchResult):
        """Save a single result to JSONL file"""
        with open(self.results_file, 'a') as f:
            f.write(json.dumps(asdict(result)) + "\n")

    def prove_theorem(self, example: Dict[str, str]) -> ProofSearchResult:
        """Prove a single theorem"""

        searcher = ProofSearch(api_client=self.api_client)

        # Setup
        try:
            repo = LeanGitRepo(example['url'], example['commit'])
            theorem = Theorem(repo, example['file_path'], example['full_name'])
        except Exception as e:
            print(f"{example['full_name']}: failed to setup theorem: {e}")
            return ProofSearchResult(
                success = False,
                theorem_name = example['full_name'],
            )

        # Run proof search
        try:
            with Dojo(theorem) as (dojo, initial_state):
                return searcher.search(theorem, dojo, initial_state)
        except DojoInitError as e:
            print(f"{example['full_name']}: DojoInitError")
            return ProofSearchResult(
                success=False,
                theorem_name=example['full_name'],
            )
        except DojoCrashError as e:
            print(f"{example['full_name']}: DojoCrashError")
            return ProofSearchResult(
                success=False,
                theorem_name=example['full_name'],
            )
        except Exception as e:
            print(f"{example['full_name']}: unknown error: {e}")
            return ProofSearchResult(
                success=False,
                theorem_name=example['full_name'],
            )

    def compute_summary(self) -> Dict:
        """Compute summary stats from results"""
        results = []
        with open(self.results_file) as f:
            for line in f:
                results.append(json.loads(line))

        total = len(results)
        successful = sum(1 for r in results if r["success"])
        failed = total - successful
        proof_lengths = [r["proof_length"] for r in results if r["success"]]
        search_times = [r["search_time"] for r in results if r["success"]]

        summary = {
            "model": self.config.model,
            "total_theorems": total,
            "successful": successful,
            "failed": failed,
            "accuracy": successful / total if total > 0 else 0.0,
            "avg_proof_length": sum(proof_lengths) / len(proof_lengths) if proof_lengths else 0.0,
            "avg_search_time": sum(search_times) / len(search_times) if search_times else 0.0,
        }

        print(f"Accuracy: {summary['accuracy']:.2%}")
        return summary

    def evaluate(self, example_limit: Optional[int] = None):
        """Run full evaluation"""
        examples = self.load_dataset(limit=example_limit)
        print(f"Starting evaluation of {len(examples)} theorems")

        # Run evaluation with parallelization
        submitted = set()
        completed_count = 0
        total = len(examples)
        executor = ProcessPoolExecutor(max_workers=self.config.num_workers)

        try:
            # Submit all + avoid dupes
            submission_to_example = {}
            for ex in examples:
                example_id = ex['full_name']
                if example_id not in submitted:
                    submission_to_example[executor.submit(self.prove_theorem, ex)] = ex
                    submitted.add(example_id)
                
            # Process results as they complete
            for submission in as_completed(submission_to_example):
                try:
                    result = submission.result()
                    self.save_result(result)
                    completed_count += 1
                    print(f"{completed_count}/{total} theorems completed")
                except Exception as e:
                    example = submission_to_example[submission]
                    print(f"Failed to process {example['full_name']}: {e}")
                    completed_count += 1
        
        except KeyboardInterrupt:
            for submission in submission_to_example:
                submission.cancel()
            executor.shutdown(wait=False, cancel_futures=True)
            print("Shutdown complete")
            raise

        finally:
            executor.shutdown(wait=True)
            print("Shutdown complete")

        # Summarize
        summary = self.compute_summary()
        with open(self.summary_file, 'w') as f:
            json.dump(summary, f, indent=2)

        print(f"Evaluation complete, results @ {self.output_path}")