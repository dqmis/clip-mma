import csv
import glob
from pathlib import Path
from typing import Callable

import torch
from PIL import Image


class StanfordCars(torch.utils.data.Dataset):
    def __init__(self, root_path: str, train: bool = True, transforms: Callable | None = None) -> None:
        subdirs_path = Path("car_data/car_data/")
        subpath = subdirs_path / "train" if train else subdirs_path / "test"

        self.root_path = Path(root_path) / "stanford-cars"
        self.split_path = self.root_path / subpath
        self.is_train_split = train
        self.transforms = transforms

        self.images = glob.glob(str(self.split_path / "**/*.jpg"), recursive=True)

        self.labels = self._extract_labels()
        self.label_map = self._build_label_map()

    def _extract_labels(self) -> list[str]:
        with open(self.root_path / "names.csv", "r") as f:
            return [i.strip().split(",")[-1] for i in f.read().splitlines()]

    def _build_label_map(self) -> dict[str, int]:
        annotations_file = f"anno_{'train' if self.is_train_split else 'test'}.csv"
        labels_map = {}

        with open(self.root_path / annotations_file, "r") as f:
            reader = csv.reader(f)
            labels_map = {row[0]: int(row[-1]) - 1 for row in reader}

        return labels_map

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, index: int) -> tuple[Image.Image, int]:
        image_file = self.images[index]
        with Image.open(image_file) as img:
            image = img.convert("RGB")
            if self.transforms:
                image = self.transforms(image)
        label = self.label_map[Path(image_file).name]

        return image, label
