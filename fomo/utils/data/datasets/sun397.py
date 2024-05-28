import json
from pathlib import Path
from typing import Any, Callable, Optional, Tuple

import requests
import os
from PIL import Image
from torchvision.datasets import SUN397 as _SUN397


class SUN397(_SUN397):

    _SPLIT_URL = "https://drive.usercontent.google.com/download?id=1y2RD81BYuiyvebdN-JymPfyWYcd8_MUq&export=download&authuser=0"

    def _download_split(self) -> dict[str, list[int]]:
        resp = requests.get(self._SPLIT_URL)
        return json.loads(resp.text)

    def __init__(
        self,
        root: str,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        split: str = "train",
        download: bool = False,
    ) -> None:
        super().__init__(root=root, transform=transform, target_transform=target_transform, download=download)
        self.split = split
        self.root = root

        self._data_dir = Path(self.root) / "SUN397"
        self._split_dict = self._download_split()

        if download:
            self._download()

        if not self._check_exists():
            raise RuntimeError("Dataset not found. You can use download=True to download it")

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
        image = Image.open(os.path.join(self.root, "SUN397", image_file)).convert("RGB")

        if self.transform:
            image = self.transform(image)

        if self.target_transform:
            label = self.target_transform(label)

        return image, label
