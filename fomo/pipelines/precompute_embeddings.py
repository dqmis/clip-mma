import argparse
import logging
from typing import Any

import torch
from torch.utils.data import Dataset, DataLoader

import wandb
from fomo.models import Model
from fomo.models._base_model import BaseModel
from fomo.utils.data.datasets import DatasetInitializer

logging.basicConfig(level=logging.INFO)

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a model on a dataset.")
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Model name",
    )
    parser.add_argument(
        "--datasets",
        type=str,
        nargs="+",
        required=True,
        help="Dataset names",
    )
    return parser.parse_args()


class EmbeddingDataset(Dataset):
    def __init__(self, image_embeddings, targets, prompt_embeddings=None):
        self.image_embeddings = image_embeddings
        self.targets = targets
        self.prompt_embeddings = prompt_embeddings

    def __len__(self):
        return len(self.image_embeddings)

    def __getitem__(self, idx):
        item = {
            "image_embedding": torch.tensor(self.image_embeddings[idx]),
            "target": torch.tensor(self.targets[idx]),
        }
        if self.prompt_embeddings is not None:
            item["prompt_embedding"] = torch.tensor(self.prompt_embeddings[idx])
        return item


class PrecomputeEmbeddingsPipeline:
    def __init__(self, model: BaseModel, dataset_initializers: list[DatasetInitializer]) -> None:
        self._model = model
        self._dataset_initializers = dataset_initializers

    def run(self) -> None:
        for dataset_loader in self._dataset_initializers:
            zero_shot_dataset, dataset_name = (
                dataset_loader.value(train=False, transforms=self._model.transforms),
                dataset_loader.name,
            )
            dataset = zero_shot_dataset.dataset, zero_shot_dataset.labels

            dataloader: DataLoader[Any] = DataLoader(
                dataset,
                batch_size=4,
                shuffle=False,
            )

            targets = []
            image_embeddings = []
            prompt_embeddings = []
            for batch in dataloader:
                images, labels = batch

                image_embeddings.extend(self._model.encode_images(images).numpy())
                targets.extend(labels)

                # TODO: Optimize. Can precompute once for all labels and then lookup embeddings
                prompts = [f"a photo of a {label}" for label in labels]
                prompt_embeddings.extend(self._model.encode_text(prompts).numpy())

            embedding_dataset = EmbeddingDataset(image_embeddings, targets, prompt_embeddings)

            save_path = f"./data/{dataset_name}_embedding_dataset.pt"
            torch.save(embedding_dataset, save_path)

            logger.info(f"Saved embedding dataset for {dataset_name} to {save_path}")


if __name__ == "__main__":
    args = parse_args()

    model = Model.from_str(args.model)
    dataset_initializers = [DatasetInitializer.from_str(dataset) for dataset in args.datasets]

    pipeline = PrecomputeEmbeddingsPipeline(model=model, dataset_initializers=dataset_initializers)

    # wandb.init(
    #     project="fomo",
    #     config={
    #         "model": args.model,
    #         "datasets": args.datasets,
    #     },
    # )

    pipeline.run()

    # wandb.log({"status": "done"})
    # wandb.finish()
