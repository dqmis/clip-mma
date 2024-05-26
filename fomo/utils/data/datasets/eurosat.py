import json
import os
from typing import Any, Callable, Optional, Tuple

import requests
from PIL import Image
from torchvision.datasets import EuroSAT as _EuroSAT


class EuroSAT(_EuroSAT):
    _SPLIT_URL = "https://drive.usercontent.google.com/download?id=1Ip7yaCWFi0eaOFUGga0lUdVi_DDQth1o&export=download&authuser=0"

    def _download_split(self) -> dict[str, list[int]]:
        resp = requests.get(self._SPLIT_URL)
        return json.loads(resp.text)

    def __init__(
        self,
        root: str,
        split: str = "train",
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        download: bool = False,
    ) -> None:
        super().__init__(root=root, transform=transform, target_transform=target_transform, download=download)

        self._split_dict = self._download_split()

        self.root = os.path.expanduser(root)
        self._base_folder = os.path.join(self.root, "eurosat")
        self._data_folder = os.path.join(self._base_folder, "2750")

        if download:
            self.download()

        if not self._check_exists():
            raise RuntimeError("Dataset not found. You can use download=True to download it")
        self.root = os.path.expanduser(root)

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

    def __getitem__(self, idx: int) -> Tuple[Any, Any]:
        image_file, label = self.data[idx], self.targets[idx]
        image = Image.open(image_file).convert("RGB")

        if self.transform:
            image = self.transform(image)

        if self.target_transform:
            label = self.target_transform(label)

        return image, label
