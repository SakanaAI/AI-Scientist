from abc import ABC, abstractmethod
from typing import List, Optional

from PIL import Image

from ab_mcts_arc2.data_types import Action
from ab_mcts_arc2.llm_generation_interface import GenerationRequest, GenerationResult
from ab_mcts_arc2.eval_result import EvalResult


class PromptTemplate(ABC):
    """
    Prompt Template for each task.
    The system prompt will **NOT** be handled by this class, and
    configuring system prompt is the responsibility of Model class constructor.
    """

    @abstractmethod
    def initial_prompt(self) -> str | List[str | Image.Image]:
        """
        Initial instruciton to be given to LLM.
        """
        raise NotImplementedError()

    @abstractmethod
    def feedback_prompt(
        self,
        action: Action,
        eval_results: Optional[List[EvalResult]],
        generation_result: GenerationResult,
    ) -> str:
        """
        Given the evaluation result, tell LLM what the result was.
        """
        raise NotImplementedError()

    @abstractmethod
    def add_next_action_instruction(
        self, action: Action, next_prompt: GenerationRequest
    ) -> GenerationRequest:
        """
        Instruct LLMs to perform what action should be performed next.
        We can append and/or modify next_prompt e.g. appending
        the action instruction at the end of it.
        """
        raise NotImplementedError()
