from abc import abstractmethod


class Model:
    @abstractmethod
    def generate(self, requests: list[str]) -> list[str]:
        raise NotImplementedError()
