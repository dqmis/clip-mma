import json
import os
from typing import Any, Callable, Optional, Union

import requests
from PIL import Image
from torchvision.datasets import Caltech101 as _Caltech101
from torchvision.datasets.utils import verify_str_arg


class Caltech101(_Caltech101):
    _SPLIT_URL = "https://drive.usercontent.google.com/download?id=1hyarUivQE36mY6jSomru6Fjd-JzwcCzN&export=download&authuser=0"

    def _download_split(self) -> dict[str, list[int]]:
        resp = requests.get(self._SPLIT_URL)
        return json.loads(resp.text)

    def __init__(
        self,
        root: str,
        split: str = "train",
        target_type: Union[list[str], str] = "category",
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        download: bool = False,
    ) -> None:
        super().__init__(
            root=root,
            transform=transform,
            target_transform=target_transform,
            download=download,
        )
        self.split = verify_str_arg(split, "split", ("train", "test"))
        self._split_dict: dict[str, list] = self._download_split()
        os.makedirs(self.root, exist_ok=True)
        if isinstance(target_type, str):
            target_type = [target_type]
        self.target_type = [verify_str_arg(t, "target_type", ("category", "annotation")) for t in target_type]

        if download:
            self.download()

        if not self._check_integrity():
            raise RuntimeError("Dataset not found or corrupted. You can use download=True to download it")

        self._categories = dict()

        self.data = []
        self.targets = []

        for filepath, label, category in self._split_dict[self.split]:
            self.data.append(filepath)
            self.targets.append(label)
            self._categories.setdefault(label, category)

        # extract dict _categories to list categories sorted by label
        self.categories = [self._categories[label] for label in sorted(self._categories.keys())]

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: int) -> tuple[Any, Any]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where the type of target specified by target_type.
        """

        img = Image.open(
            os.path.join(
                self.root,
                "101_ObjectCategories",
                self.data[index],
            )
        )

        target = self.targets[index]

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target
