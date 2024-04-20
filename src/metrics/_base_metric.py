from abc import ABC, abstractmethod
from typing import Any

from numpy.typing import NDArray


class Metric(ABC):
    @abstractmethod
    def evaluate(self, y_true: NDArray[Any], y_pred: NDArray[Any]) -> float:
        pass
