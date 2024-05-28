import json
import os
import time
from dataclasses import dataclass
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.utils
from sklearn.manifold import TSNE

from fomo.pipelines.utils.utils import build_label_prompts

try:
    from wandb import wandb
except ImportError:
    print("Wandb is not installed. Please install it to log the results.")

import logging

from fomo.pipelines.utils.initializers import (
    initalize_dataloaders,
    initalize_datasets,
    initalize_test_dataloader_subsample,
    intialize_model,
)

logging.basicConfig(level=logging.INFO)

logger = logging.getLogger(__name__)


@dataclass
class AnalyzerArgs:
    model_checkpoint_path: str = "checkpoints/model_best.pth.tar"
    model_backbone: str = "ViT-B/16"
    model_type: str = "clip_base"

    use_wandb: bool = False

    output_dir: str = "./output/analysis/"
    dataset: str = "cifar10"
    device: str = "cuda"

    text_prompt_template: str = "a photo of {}."
    seed: int = 42

    batch_size: int = 64
    num_workers: int = 4
    train_size: float | None = None
    train_eval_size: tuple[int, int] | None = None
    train_subsample: str = "all"
    test_subsample: str = "all"

    def __post_init__(self) -> None:
        self.run_id = f"{self.model_type}_{self.model_backbone}_{str(int(time.time()))}".replace(
            "/", ""
        ).lower()
        self.output_dir = os.path.join(self.output_dir, self.run_id)

    def save_config(self) -> None:
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

        with open(os.path.join(self.output_dir, "config.json"), "w") as f:
            data = self.to_dict()
            json.dump(data, f, indent=4)

    def to_dict(self) -> dict[str, Any]:
        return self.__dict__


class Analyzer:
    """Analyze CLIP adapted features."""

    def __init__(self, analyzer_args: AnalyzerArgs) -> None:
        self._analyzer_args = analyzer_args

        # Load clip image transformation
        self.model = intialize_model(
            self._analyzer_args.model_type, self._analyzer_args.model_backbone, self._analyzer_args.device
        )

        # loading best model
        self.model.load_state_dict(
            torch.load(
                self._analyzer_args.model_checkpoint_path,
                map_location=torch.device(self._analyzer_args.device),
            )["state_dict"],
        )

        self.transforms = self.model.transforms

        (train_dataset, test_dataset), (train_labels, test_labels) = initalize_datasets(
            self._analyzer_args.dataset,
            self.transforms,
            self._analyzer_args.train_subsample,
            self._analyzer_args.test_subsample,
        )

        self.train_labels = train_labels
        self.test_labels = test_labels

        self.num_classes = len(set([*train_labels, *test_labels]))

        self.train_loader, self.val_loader, self.test_loader = initalize_dataloaders(
            train_dataset, test_dataset, self._analyzer_args
        )

    def collect_features(self, data_loader):
        self.model.eval()
        image_features_list = []
        text_features_list = []
        original_image_features_list = []
        original_text_features_list = []
        labels_list = []

        num_images = 500
        i = 0

        with torch.no_grad():
            for images, targets in data_loader:
                i += 1
                if i > num_images:
                    break

                images = images.to(self._analyzer_args.device)
                (
                    image_features,
                    text_features,
                    original_image_features,
                    original_text_features,
                ) = self.model.encode_features(images)
                image_features = image_features.squeeze(1)
                text_features = text_features.squeeze(1)

                image_features_list.append(image_features.numpy())
                text_features_list.append(text_features.numpy())
                original_image_features_list.append(original_image_features.numpy())
                original_text_features_list.append(original_text_features.numpy())
                labels_list.append(targets.numpy())

        image_features_list = np.concatenate(image_features_list)
        text_features_list = np.concatenate(text_features_list)
        original_image_features_list = np.concatenate(original_image_features_list)
        original_text_features_list = np.concatenate(original_text_features_list)
        labels_list = np.concatenate(labels_list)

        # Store to file
        np.save(os.path.join(self._analyzer_args.output_dir, "image_features.npy"), image_features_list)
        np.save(os.path.join(self._analyzer_args.output_dir, "text_features.npy"), text_features_list)
        np.save(
            os.path.join(self._analyzer_args.output_dir, "original_image_features.npy"),
            original_image_features_list,
        )
        np.save(
            os.path.join(self._analyzer_args.output_dir, "original_text_features.npy"),
            original_text_features_list,
        )
        np.save(os.path.join(self._analyzer_args.output_dir, "labels.npy"), labels_list)

        return (
            image_features_list,
            text_features_list,
            original_image_features_list,
            original_text_features_list,
            labels_list,
        )

    def run_tsne(self, features):
        print("Running t-SNE")
        print(f"Features shape: {features.shape}")
        tsne = TSNE(n_components=2, random_state=self._analyzer_args.seed, perplexity=30)  # self.num_classes
        return tsne.fit_transform(features)

    def plot_features(self, tsne_results, labels, title):
        plt.figure(figsize=(12, 10))
        for label in np.unique(labels):
            indices = labels == label
            plt.scatter(tsne_results[indices, 0], tsne_results[indices, 1], label=label)
        plt.legend()
        plt.title(title)
        plt.savefig(os.path.join(self._analyzer_args.output_dir, "tsne_plot.png"))

    def analyze(self, split: str = "valid") -> float:
        """Evaluates the model on the given `split` set and returns average accuracy."""
        print(f"Analyzing {split} set")
        if split == "valid":
            loader = self.val_loader
        else:
            test_subsample = split.split("_")[-1]
            loader, test_labels = initalize_test_dataloader_subsample(
                self._analyzer_args.dataset, self.transforms, self._analyzer_args, test_subsample
            )

            # precompute train prompt features
            self.model.precompute_prompt_features(
                build_label_prompts(test_labels, self._analyzer_args.text_prompt_template)
            )

        (
            image_features,
            text_features,
            original_image_features,
            original_text_features,
            labels,
        ) = self.collect_features(loader)

        image_tsne = self.run_tsne(image_features)
        # text_tsne = self.run_tsne(text_features)
        original_image_tsne = self.run_tsne(original_image_features)
        # original_text_tsne = self.run_tsne(original_text_features)

        # print(f"label shape: {labels.shape}")

        self.plot_features(image_tsne, labels, title="t-SNE of Adapted Image Features")
        # self.plot_features(text_tsne, labels_for_text, title="t-SNE of Adapted Text Features")
        self.plot_features(original_image_tsne, labels, title="t-SNE of Original Image Features")
        # self.plot_features(original_text_tsne, labels_for_text, title="t-SNE of Original Text Features")

    def run(self) -> None:
        """Runs training for the specified number of epochs."""

        self._analyzer_args.save_config()

        print(f"Loaded model from {self._analyzer_args.model_checkpoint_path}")

        if self._analyzer_args.use_wandb:
            wandb.init(
                project="fomo",
                name=self._analyzer_args.run_id,
                config=self._analyzer_args.to_dict(),
                reinit=True,
            )

        # # evaluate on test set
        self.analyze("test_all")
        # self.analyze("test_base")
        # self.analyze("test_new")
