from typing_extensions import Any

from numpy.typing import NDArray
from sklearn.metrics import accuracy_score, precision_score, recall_score

from fomo.metrics._base_metric import Metric


class Accuracy(Metric):
    @staticmethod
    def calculate(y_true: NDArray[Any], y_pred: NDArray[Any]) -> float:
        score: float = accuracy_score(y_true, y_pred)
        return score


class Precision(Metric):
    @staticmethod
    def calculate(y_true: NDArray[Any], y_pred: NDArray[Any]) -> float:
        score: float = precision_score(y_true, y_pred)
        return score


class Recall(Metric):
    @staticmethod
    def calculate(y_true: NDArray[Any], y_pred: NDArray[Any]) -> float:
        score: float = recall_score(y_true, y_pred)
        return score


METRICS = [Accuracy()]
