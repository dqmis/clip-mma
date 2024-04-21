from abc import ABC, abstractmethod
from typing import Any

from numpy.typing import NDArray


class Metric(ABC):

    @staticmethod
    @abstractmethod
    def calculate(y_true: NDArray[Any], y_pred: NDArray[Any]) -> float:
        pass

    def __str__(self) -> str:
        return self.__class__.__name__
