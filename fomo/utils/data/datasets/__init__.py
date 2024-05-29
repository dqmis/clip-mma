import enum
from typing import Callable, Self

from torchvision import datasets

from fomo.utils.data.datasets import (
    _labels,
    caltech101,
    eurosat,
    oxford_flowers,
    stanford_cars,
    sun397,
    ucf101,
)
from fomo.utils.data.zero_shot_dataset import ZeroShotDataset


class CIFAR10(ZeroShotDataset):
    def __init__(self, train: bool, root: str = "data", transforms: Callable | None = None) -> None:
        dataset = datasets.CIFAR10(root=root, train=train, download=True, transform=transforms)

        super().__init__(dataset=dataset)

    @property
    def labels(self) -> list[str]:
        return _labels.CIFAR10


class Caltech101(ZeroShotDataset):
    def __init__(self, train: bool, root: str = "data", transforms: Callable | None = None) -> None:
        dataset = caltech101.Caltech101(
            root=root, split="train" if train else "test", download=True, transform=transforms
        )

        super().__init__(dataset=dataset)

    @property
    def labels(self) -> list[str]:
        return [" ".join(val.lower().split("_")) for val in self.dataset.categories]  # type: ignore


class OxfordFlowers(ZeroShotDataset):
    def __init__(self, train: bool, root: str = "data", transforms: Callable | None = None) -> None:
        dataset = oxford_flowers.OxfordFlowers(
            root=root, split="train" if train else "test", download=True, transform=transforms
        )

        super().__init__(dataset=dataset)

    @property
    def labels(self) -> list[str]:
        return _labels.OXFORD_FLOWERS


class OxfordPets(ZeroShotDataset):
    def __init__(self, train: bool, root: str = "data", transforms: Callable | None = None) -> None:
        dataset = datasets.OxfordIIITPet(
            root=root, split="trainval" if train else "test", download=True, transform=transforms
        )

        super().__init__(dataset=dataset)

    @property
    def labels(self) -> list[str]:
        return self.dataset.classes  # type: ignore


class Food101(ZeroShotDataset):
    def __init__(self, train: bool, root: str = "data", transforms: Callable | None = None) -> None:
        dataset = datasets.Food101(
            root=root, download=True, split="train" if train else "test", transform=transforms
        )

        super().__init__(dataset=dataset)

    @property
    def labels(self) -> list[str]:
        return [val.replace("_", " ") for val in self.dataset.classes]  # type: ignore


class StanfordCars(ZeroShotDataset):
    def __init__(self, train: bool, root: str = "data", transforms: Callable | None = None) -> None:
        dataset = stanford_cars.StanfordCars(root_path=root, train=train, transforms=transforms)

        super().__init__(dataset=dataset)

    @property
    def labels(self) -> list[str]:
        return self.dataset.labels  # type: ignore


class FGVCAircraft(ZeroShotDataset):
    def __init__(self, train: bool, root: str = "data", transforms: Callable | None = None) -> None:
        dataset = datasets.FGVCAircraft(
            root=root, download=True, split="trainval" if train else "test", transform=transforms
        )

        super().__init__(dataset=dataset)

    @property
    def labels(self) -> list[str]:
        return self.dataset.classes  # type: ignore


class Imagenet(ZeroShotDataset):
    def __init__(self, train: bool, root: str = "data", transforms: Callable | None = None) -> None:
        dataset = datasets.ImageNet(
            root=root, split="train" if train else "val", download=True, transform=transforms
        )

        super().__init__(dataset=dataset)

    @property
    def labels(self) -> list[str]:
        return self.dataset.class_to_idx.keys()  # type: ignore


class SUN397(ZeroShotDataset):
    def __init__(self, train: bool, root: str = "data", transforms: Callable | None = None) -> None:
        dataset = sun397.SUN397(
            root=root, download=True, split="train" if train else "test", transform=transforms
        )

        super().__init__(dataset=dataset)

    @property
    def labels(self) -> list[str]:
        return self.dataset.categories  # type: ignore


class DTD(ZeroShotDataset):
    def __init__(self, train: bool, root: str = "data", transforms: Callable | None = None) -> None:
        dataset = datasets.DTD(
            root=root, download=True, split="train" if train else "test", transform=transforms
        )

        super().__init__(dataset=dataset)

    @property
    def labels(self) -> list[str]:
        return self.dataset.classes  # type: ignore


class EuroSAT(ZeroShotDataset):
    def __init__(self, train: bool, root: str = "data", transforms: Callable | None = None) -> None:
        dataset = eurosat.EuroSAT(
            root=root, download=True, split="train" if train else "test", transform=transforms
        )

        super().__init__(dataset=dataset)

    @property
    def labels(self) -> list[str]:
        return _labels.EUROSAT


class UCF101(ZeroShotDataset):
    def __init__(self, train: bool, root: str = "data", transforms: Callable | None = None) -> None:
        dataset = ucf101.UCF101(root=root, train=train, transform=transforms, download=True)

        super().__init__(dataset=dataset)

    @property
    def labels(self) -> list[str]:
        return [val.replace("_", " ").lower() for val in self.dataset.categories]  # type: ignore


PROMPTS = {
    "cifar10": "a photo of a {}.",
    "imagenet": "a photo of a {}.",
    "caltech101": "a photo of a {}.",
    "oxford_pets": "a photo of a {}, a type of pet.",
    "oxford_flowers": "a photo of a {}, a type of flower.",
    "food101": "a photo of {}, a type of food.",
    "stanford_cars": "a photo of a {}.",
    "fgvc_aircraft": "a photo of a {}, a type of aircraft.",
    "sun397": "a photo of a {}.",
    "dtd": "{} texture.",
    "eurosat": "a centered satellite photo of {}.",
    "ucf101": "a photo of a person doing {}.",
}


class DatasetInitializer(enum.Enum):
    CIFAR10 = CIFAR10
    IMAGENET = Imagenet
    STANFORD_CARS = StanfordCars
    OXFORD_FLOWERS = OxfordFlowers
    OXFORD_PETS = OxfordPets
    FOOD101 = Food101
    CALTECH101 = Caltech101
    FGVC_AIRCRAFT = FGVCAircraft
    SUN397 = SUN397
    DTD = DTD
    EUROSAT = EuroSAT
    UCF101 = UCF101

    @classmethod
    def from_str(cls, name: str) -> Self:
        return cls[name.upper()]
