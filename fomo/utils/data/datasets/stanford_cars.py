import os

import torch
from PIL import Image


class StanfordCars(torch.utils.data.Dataset):
    def __init__(self, root_path: str, train: bool = True) -> None:
        root_path = os.path.join(root_path, "stanford-cars")
        subdirs_path = "car_data/car_data/"
        subpath = subdirs_path + "train" if train else subdirs_path + "test"
        self.root_path = root_path
        self.split_path = os.path.join(root_path, subpath)

        self.images = [
            os.path.join(self.split_path, subdir, file)
            for subdir in os.listdir(self.split_path)
            for file in os.listdir(os.path.join(self.split_path, subdir))
            if file.endswith(".jpg")
        ]

        self.labels = self._extract_labels()
        self.is_train_split = train
        self.label_map = self._build_label_map()

    def _extract_labels(self) -> list[str]:
        with open(os.path.join(self.root_path, "names.csv"), "r") as f:
            return [i.strip().split(",")[-1] for i in f.read().splitlines()]

    def _build_label_map(self) -> dict[str, int]:
        annotations_file = f"anno_{'train' if self.is_train_split else 'test'}.csv"

        labels_map = {}

        with open(os.path.join(self.root_path, annotations_file), "r") as f:
            samples = [i.strip().split(",") for i in f.read().splitlines()]
        for sample in samples:
            labels_map[sample[0]] = int(sample[-1]) - 1

        return labels_map

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, index: int) -> tuple[Image.Image, int]:
        image_file = self.images[index]
        image = Image.open(image_file).convert("RGB")
        return image, self.label_map[image_file.split("/")[-1]]
