import json
from pathlib import Path
from typing import List, Optional, Tuple

from ab_mcts_arc2.data_types import Action, ARCProbData, Grid
from ab_mcts_arc2.llm_generation_interface import GenerationResult
from ab_mcts_arc2.eval_result import EvalResult, EvalResultWithAns
from ab_mcts_arc2.evaluate_code import eval_if_test_pass
from ab_mcts_arc2.tasks.base import Task


class ARCProblem(Task):
    def __init__(self, problem: ARCProbData, label: Optional[str] = None) -> None:
        self.demos = problem["train"]
        self.tests = problem["test"]

        self.label = label

    @classmethod
    def load_file(cls, json_path: Path | str) -> "ARCProblem":
        prob_path = Path(json_path)
        if not prob_path.exists():
            raise RuntimeError(f"ARC problem not found at {str(prob_path)}")

        return cls(problem=json.loads(prob_path.read_text()), label=prob_path.stem)

    def generate_eval_results(
        self, llm_answer: GenerationResult, kind: Action
    ) -> Optional[List[EvalResult]]:
        llm_answer_code = llm_answer.parse_python_code()
        if llm_answer_code is None:
            return None

        if kind == "transform":
            results = self.run_transform_on_demos(llm_answer_code)
        else:
            raise NotImplementedError()

        return results

    def evaluate_on_test(
        self, llm_answer: GenerationResult
    ) -> Tuple[List[EvalResult], float]:
        py_code = llm_answer.parse_python_code()
        if py_code is None:
            return [], 0.0

        eval_results = self.run_transform_on_tests_and_check(py_code)
        score = (
            1.0
            if all(eval_result.get_score() == 1.0 for eval_result in eval_results)
            else 0.0
        )
        return eval_results, score

    def run_transform_on_demos(self, transform_code: str) -> List[EvalResult]:
        eval_results: List[EvalResultWithAns] = []
        for demo in self.demos:
            is_correct, output = eval_if_test_pass(demo, transform_code, "transform")
            eval_results.append(
                EvalResultWithAns(answer=output, groundtruth=demo["output"])
            )
        return eval_results

    def run_transform_on_tests_and_check(self, transform_code: str) -> List[EvalResult]:
        eval_results: List[EvalResultWithAns] = []
        for test in self.tests:
            is_correct, output = eval_if_test_pass(test, transform_code, "transform")
            eval_results.append(
                EvalResultWithAns(answer=output, groundtruth=test["output"])
            )
        return eval_results

    def run_transform_on_tests(self, transform_code: str) -> List[EvalResult]:
        """
        Does not check if it is correct on test
        """
        eval_results: List[EvalResultWithAns] = []
        for test in self.tests:
            _is_correct, output = eval_if_test_pass(test, transform_code, "transform")
            eval_results.append(EvalResultWithAns(answer=output, groundtruth=None))
        return eval_results

    def check_on_tests(self, preds: List[Grid]) -> bool:
        """True if correct on all tests"""
        for pred, test in zip(preds, self.tests):
            if pred != test["output"]:
                return False
        return True
