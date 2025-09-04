import math
from dataclasses import dataclass

import treequest as tq

from ab_mcts_arc2.llm_generation_interface import GenerationResult
from ab_mcts_arc2.eval_result import EvalResult
from ab_mcts_arc2.tasks.arc.task import ARCProblem


def is_power_of_two(n: int):
    return n > 0 and (n & (n - 1)) == 0


def get_top_k(state, algorithm, k):
    if hasattr(state, "tree"):
        # nodes in ascending order
        nodes = state.tree.get_nodes()
        nodes.sort(key=lambda x: x.score, reverse=True)
        top_k = [(node.state, node.score) for node in nodes[:k]]
        return top_k
    else:
        return tq.top_k(state, algorithm, k=k)


@dataclass
class NodeState:
    generation_result: GenerationResult
    eval_results: EvalResult
    model_name: str


def calculate_is_correct(
    state, algorithm: tq.Algorithm, task: ARCProblem, k: int
) -> bool:
    for node_state, public_score in get_top_k(state, algorithm, k=k):
        eval_results, _score = task.evaluate_on_test(
            llm_answer=node_state.generation_result
        )

        if len(eval_results) == 0:
            private_score = 0
        else:
            private_score = sum(
                [eval_result.get_score() for eval_result in eval_results]
            ) / len(eval_results)

        if math.isclose(
            private_score,
            1,
        ):
            return True

    return False
