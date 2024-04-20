from typing import Any

from torch.utils.data import Dataset

from src.metrics._base_metric import Metric
from src.models._base_model import BaseModel


class EvaluatePipeline:
    def __init__(self, model: BaseModel, datasets: list[Dataset[Any]], metrics: list[Metric]) -> None:
        self._model = model
        self._datasets = datasets
        self._metrics = metrics

    def run(self):
        return {metric: metric(self.model) for metric in self.metrics}
