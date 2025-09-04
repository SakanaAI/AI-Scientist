from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Literal, Optional, Sequence

# NOTE: assistant_prefill is used for prefilling the assistant's response
# and is specific to fishfarm, e.g., for the Evalplus task we might want to
# add a prefix like "```python" so that the models generates only python code
# It is also applicable to chain-of-thought trick where we want to use
# "Let's think step by step." as the prefill text.
# see https://github.com/SakanaAI/fishfarm/issues/58
Role = Literal["system", "user", "assistant", "assistant_prefill"]


@dataclass
class Message:

    role: Role
    content: str


@dataclass
class GenerationRequest:

    messages: list[Message]

    # TODO: are they necessary?
    max_tokens: Optional[int] = None
    stop: Sequence[str] = ()


@dataclass
class GenerationResult:

    request: GenerationRequest
    generation: str


@dataclass
class NLLRequest:

    messages: list[Message]


@dataclass
class NLLResult:

    request: NLLRequest
    sum_nll: float
    num_considered_tokens: int


class Model:

    def generate(
        self, requests: Sequence[GenerationRequest]
    ) -> Iterable[GenerationResult]:
        raise NotImplementedError()

    def nll(self, requests: Sequence[NLLRequest]) -> Iterable[NLLResult]:
        raise NotImplementedError()
