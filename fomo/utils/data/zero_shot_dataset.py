from abc import ABC, abstractmethod

from torch.utils.data import Dataset


class ZeroShotDataset(ABC):
    def __init__(self, dataset: Dataset) -> None:
        self._dataset = dataset

    @property
    def dataset(self) -> Dataset:
        return self._dataset

    @property
    @abstractmethod
    def labels(self) -> list[str]:
        pass
