import time
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, List, Optional, Set
from lean_dojo import Dojo, Theorem, TacticState, ProofFinished, LeanError, ProofGivenUp

from benchmarking.api_clients import APIClient


@dataclass
class SearchNode:
    """Node in the proof search tree."""
    state: TacticState
    depth: int
    tactic_sequence: List[str] = field(default_factory=list) # ensure new empty list


@dataclass
class ProofSearchResult:
    """Result of a proof search attempt"""
    success: bool
    theorem_name: str
    proof_steps: Optional[List[str]] = None
    proof_length: Optional[int] = None
    search_time: float = 0.0


class ProofSearch:
    """Proof search using generated tactics"""

    def __init__(self, api_client: APIClient):
        self.api_client = api_client
        self.max_expansions = 500
        self.max_depth = 50
        self.timeout = 300

    def search(self, theorem: Theorem, dojo: Dojo, initial_state: TacticState) -> ProofSearchResult:
        """Do the search"""
        start_time = time.time()
        theorem_name = theorem.full_name
        print(f"{theorem_name}: starting search")

        # Initialize
        queue: Deque[SearchNode] = deque()
        visited: Set[str] = set()
        num_expansions = 0
        queue.append(SearchNode(
            state=initial_state,
            depth=0,
            tactic_sequence=[]
        ))

        # Search
        while queue and num_expansions < self.max_expansions:
            # Check for timeout
            elapsed_time = time.time() - start_time
            if elapsed_time > self.timeout:
                print(f"{theorem_name}: proof search timed out")
                return ProofSearchResult(
                    success = False,
                    theorem_name = theorem_name,
                    search_time = elapsed_time,
                )

            node = queue.popleft()
            state_pp = node.state.pp

            if state_pp in visited: # already visited
                continue
            visited.add(state_pp)
            if node.depth >= self.max_depth: # depth limit
                continue

            num_expansions += 1

            # Generate tactics
            try:
                suggestions = self.api_client.generate_tactics(state_pp)
            except Exception as e:
                print(f"{theorem_name}: generation failed: {e}")
                continue

            # Try each suggested tactic
            for suggestion in suggestions:
                try:
                    result = dojo.run_tac(node.state, suggestion)

                    if isinstance(result, ProofFinished):
                        proof_steps = node.tactic_sequence + [suggestion]
                        elapsed_time = time.time() - start_time
                        print(f"{theorem_name}: PROVED")
                        return ProofSearchResult(
                            success = True,
                            theorem_name = theorem_name,
                            proof_steps = proof_steps,
                            proof_length = len(proof_steps),
                            search_time = elapsed_time
                        )

                    elif isinstance(result, TacticState):
                        if result.pp in visited: # already visisted
                            continue
                        queue.append(SearchNode(
                            state = result,
                            depth = node.depth+1,
                            tactic_sequence = node.tactic_sequence + [suggestion]
                        ))

                    elif isinstance(result, LeanError):
                        # print("LeanError")
                        continue

                    elif isinstance(result, ProofGivenUp):
                        # print("ProofGivenUp")
                        continue

                except Exception as e:
                    print(f"{theorem_name}: run_tac failed: {e}")
                    continue

        # Search exhausted
        elapsed_time = time.time() - start_time
        print(f"{theorem_name}: search exhausted")
        return ProofSearchResult(
            success = False,
            theorem_name = theorem_name,
            search_time = elapsed_time,
        )