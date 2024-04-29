from abc import ABC, abstractmethod
from typing import Any

from numpy.typing import NDArray
from torch.utils.data import DataLoader
from torchvision.transforms import Compose


class BaseModel(ABC):
    @abstractmethod
    def predict_for_eval(self, x: DataLoader[Any]) -> tuple[NDArray[Any], NDArray[Any]]:
        """Predicts the output of the model for evaluation purposes (y_true, y_pred)."""

        pass

    @property
    @abstractmethod
    def transforms(self) -> Compose:
        """The transforms used for inference and training."""

        pass

    @property
    @abstractmethod
    def batch_size(self) -> int:
        """The batch size used for inference and training."""

        pass

    @abstractmethod
    def reconfig_labels(
        self,
        labels: list[str],
    ) -> None:
        """Reconfigures the model for a given dataset."""

        pass
