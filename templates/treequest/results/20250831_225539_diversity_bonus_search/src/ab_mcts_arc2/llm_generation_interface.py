import re
from abc import abstractmethod
from dataclasses import dataclass
from typing import Iterable, List, Optional

from ab_mcts_arc2.model_base import GenerationRequest as BaseGenerationRequest
from ab_mcts_arc2.model_base import GenerationResult as BaseGenerationResult
from ab_mcts_arc2.model_base import Message as BaseMessage
from ab_mcts_arc2.model_base import Model as BaseModel
from PIL import Image


class GenerationResult(BaseGenerationResult):
    def parse_python_code(self) -> Optional[str]:
        code = re.search(
            r"```python\n(.*)\n```", self.generation, re.DOTALL | re.MULTILINE
        )
        if code is None:
            return None
        return code.group(1).replace("\t", " " * 4)

    def parse_score_block(self) -> Optional[str]:
        score = re.search(r"```score(.*)```", self.generation, re.DOTALL | re.MULTILINE)
        if score is None:
            return None
        return score.group(1).strip()


class GenerationRequest(BaseGenerationRequest):
    def get_last_generation_result(self) -> GenerationResult:
        assert self.messages[-1].role == "user" and len(self.messages) >= 3
        assert self.messages[-2].role == "assistant"

        return GenerationResult(
            request=GenerationRequest(messages=self.messages[:-2]),
            generation=self.messages[-2].content,
        )


class Model(BaseModel):

    @abstractmethod
    def generate(
        self, requests: Image.Sequence[GenerationRequest]
    ) -> Iterable[GenerationResult]:
        raise NotImplementedError()


@dataclass
class Message(BaseMessage):
    content: str | List[str | Image.Image]
