import json
from pathlib import Path
from typing import Callable

import requests
from torchvision.datasets import Flowers102
from torchvision.datasets.utils import verify_str_arg


class OxfordFlowers(Flowers102):
    _SPLIT_URL = "https://drive.usercontent.google.com/download?id=1Pp0sRXzZFZq15zVOzKjKBu4A9i01nozT&export=download&authuser=0"

    def _download_split(self) -> dict[str, list[int]]:
        resp = requests.get(self._SPLIT_URL)
        return json.loads(resp.text)

    def __init__(
        self,
        root: str,
        split: str = "train",
        transform: Callable | None = None,
        target_transform: Callable | None = None,
        download: bool = False,
    ) -> None:
        super().__init__(root, transform=transform, target_transform=target_transform)
        self._split = verify_str_arg(split, "split", ("train", "val", "test"))
        self._base_folder = Path(self.root) / "flowers-102"
        self._images_folder = self._base_folder / "jpg"
        self._split_dict = self._download_split()

        if download:
            self.download()

        if not self._check_integrity():
            raise RuntimeError("Dataset not found or corrupted. You can use download=True to download it")

        self._labels = []
        self._image_files = []

        for image_id, image_label, _ in self._split_dict[self._split]:
            self._labels.append(image_label)
            self._image_files.append(self._images_folder / image_id)
