import json
import os
import ssl
from pathlib import Path
from typing import Any, Callable, Optional, Tuple

import requests
from PIL import Image
from torch.utils.data import Dataset
from torchvision.datasets import UCF101 as _UCF101
from torchvision.datasets import VisionDataset
from torchvision.datasets.utils import download_and_extract_archive


class UCF101(VisionDataset):
    _SPLIT_URL = "https://drive.usercontent.google.com/download?id=1I0S0q91hJfsV9Gf4xDIjgDq4AqBNJb1y&export=download&authuser=0"
    _DATASET_URL = "https://drive.usercontent.google.com/download?id=10Jqome3vtUA2keJkNanAiFpgbyC9Hc2O&export=download&authuser=0&confirm=t&uuid=7543a322-e294-4c7a-9b80-172e644f0b02&at=APZUnTWmj4xoWwpIk0N_qZS5heJF%3A1717005803820"

    def _download_split(self) -> dict[str, list[int]]:
        resp = requests.get(self._SPLIT_URL)
        return json.loads(resp.text)

    def __init__(
        self,
        root: str,
        train: bool = True,
        transform: Optional[Callable] = None,
        download: bool = False,
    ) -> None:
        split = "train" if train else "test"

        self.transform = transform
        self.split = split
        self._split_dict = self._download_split()

        self.root = os.path.expanduser(root)
        self._data_dir = Path(self.root, "UCF-101-midframes")

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
        image = Image.open(os.path.join(self._data_dir, image_file)).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, label

    def _check_exists(self) -> bool:
        return self._data_dir.is_dir()

    def download(self) -> None:
        if self._check_exists():
            return
        download_and_extract_archive(self._DATASET_URL, download_root=self.root, filename="ucf101.zip")
