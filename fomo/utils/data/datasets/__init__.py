import enum
from typing import Self

from torchvision import datasets

from fomo.utils.data.datasets import _labels, stanford_cars
from fomo.utils.data.zero_shot_dataset import ZeroShotDataset


class CIFAR10(ZeroShotDataset):
    def __init__(self, train: bool, root: str = "data") -> None:
        dataset = datasets.CIFAR10(root=root, train=train, download=True)

        super().__init__(dataset=dataset)

    @property
    def labels(self) -> list[str]:
        return _labels.CIFAR10


class OxfordFlowers(ZeroShotDataset):
    def __init__(self, train: bool, root: str = "data") -> None:
        dataset = datasets.Flowers102(root=root, split="train" if train else "test", download=True)

        super().__init__(dataset=dataset)

    @property
    def labels(self) -> list[str]:
        return _labels.OXFORD_FLOWERS


class OxfordPets(ZeroShotDataset):
    def __init__(self, train: bool, root: str = "data") -> None:
        dataset = datasets.OxfordIIITPet(root=root, split="trainval" if train else "test", download=True)

        super().__init__(dataset=dataset)

    @property
    def labels(self) -> list[str]:
        return self.dataset.classes  # type: ignore


class Food101(ZeroShotDataset):
    def __init__(self, train: bool, root: str = "data") -> None:
        dataset = datasets.Food101(root=root, download=True, split="train" if train else "test")

        super().__init__(dataset=dataset)

    @property
    def labels(self) -> list[str]:
        return [val.replace("_", " ") for val in self.dataset.classes]  # type: ignore


class StanfordCars(ZeroShotDataset):
    def __init__(self, train: bool, root: str = "data") -> None:
        dataset = stanford_cars.StanfordCars(root_path=root, train=train)

        super().__init__(dataset=dataset)

    @property
    def labels(self) -> list[str]:
        return self.dataset.labels  # type: ignore


class DatasetInitializer(enum.Enum):
    CIFAR10 = CIFAR10
    STANFORD_CARS = StanfordCars
    OXFORD_FLOWERS = OxfordFlowers
    OXFORD_PETS = OxfordPets
    FOOD101 = Food101

    @classmethod
    def from_str(cls, name: str) -> Self:
        return cls[name.upper()]
