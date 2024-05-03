import argparse
import logging
from typing import Any


from torch.optim import Adam
from torch.utils.data import DataLoader, random_split

import wandb
from fomo.models import Model
from fomo.models._base_model import BaseModel
from fomo.utils.data.datasets import CIFAR10, DatasetInitializer
from fomo.training.trainer import Trainer


logging.basicConfig(level=logging.INFO)

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a model on a dataset.")
    parser.add_argument("--model", type=str, required=True, help="Model name")
    parser.add_argument("--dataset", type=str, nargs="+", required=True, help="Dataset names")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=1e-5, help="Learning rate")
    parser.add_argument("--val_split", type=float, default=0.1, help="Fraction of data to use for validation")
    return parser.parse_args()


class TrainPipeline:
    def __init__(
        self,
        model: BaseModel,
        trainable_parameters: Any,  # TODO
        dataset_initializers: list[DatasetInitializer],
        epochs: int,
        lr: float,
        val_split: float,
    ) -> None:
        self._model = model
        self._trainable_parameters = trainable_parameters
        self._dataset_initializers = dataset_initializers
        self._epochs = epochs
        self._lr = lr
        self._val_split = val_split

    def run(self) -> None:
        optimizer = Adam(self._trainable_parameters, lr=self._lr)

        for dataset_loader in self._dataset_initializers:  # TODO: Should only train on one dataset
            train_dataset = dataset_loader.value(train=True, transforms=self._model.transforms).dataset
            # val_size = int(len(dataset) * self._val_split)
            # train_size = len(dataset) - val_size
            # train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

            train_dataloader = DataLoader(train_dataset, batch_size=self._model.batch_size, shuffle=True)
            # val_dataloader = DataLoader(val_dataset, batch_size=self._model.batch_size, shuffle=False)

            logger.info(f"Training model on dataset {train_dataset.__class__.__name__}")
            trainer = Trainer(
                model=self._model,
                optimizer=optimizer,
                train_dataloader=train_dataloader,
                val_dataloader=train_dataloader,  # TODO: Change to val_dataloader
            )
            trainer.train(epochs=self._epochs)


if __name__ == "__main__":
    args = parse_args()

    model = Model.from_str(args.model)
    dataset_initializers = [DatasetInitializer.from_str(dataset) for dataset in args.datasets]

    pipeline = TrainPipeline(
        model=model,
        trainable_parameters=model._clip.parameters(),
        dataset_initializers=dataset_initializers,
        epochs=args.epochs,
        lr=args.lr,
        val_split=args.val_split,
    )

    wandb.init(
        project="fomo",
        config={
            "model": args.model,
            "datasets": args.datasets,
            "epochs": args.epochs,
            "lr": args.lr,
            "val_split": args.val_split,
        },
    )

    pipeline.run()

    wandb.finish()
