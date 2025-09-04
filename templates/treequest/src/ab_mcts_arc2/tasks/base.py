from abc import ABC
from typing import List, Optional, Tuple

from ab_mcts_arc2.data_types import Action
from ab_mcts_arc2.llm_generation_interface import GenerationResult
from ab_mcts_arc2.eval_result import EvalResult


class Task(ABC):
    def generate_eval_results(
        self, llm_answer: GenerationResult, kind: Action
    ) -> Optional[List[EvalResult]]:
        raise NotImplementedError()

    def evaluate_on_test(
        self, llm_answer: GenerationResult
    ) -> Tuple[List[EvalResult], float]:
        raise NotImplementedError()
