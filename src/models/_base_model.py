from abc import ABC, abstractmethod
from typing import Any

from numpy.typing import NDArray
from torch.utils.data import DataLoader


class BaseModel(ABC):
    @abstractmethod
    def predict(self, x: DataLoader) -> NDArray[Any]:
        pass
