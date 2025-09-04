from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Generic, Optional, TypeVar, Dict, Any

T = TypeVar("T")


@dataclass
class EvalResult(ABC):
    @abstractmethod
    def get_score(self) -> float:
        raise NotImplementedError()


@dataclass
class EvalResultWithAns(EvalResult, Generic[T]):
    answer: T
    groundtruth: T

    def get_score(self) -> float:
        return 1 if self.answer == self.groundtruth else 0


@dataclass
class EvalResultLiveCodeBench(EvalResultWithAns):
    score: float
    additional_info: Dict[str, Any]

    def get_score(self) -> float:
        return self.score


@dataclass
class EvalResultWithScore(EvalResult):
    score: float
    reason: Optional[str] = None

    def get_score(self) -> float:
        return self.score
