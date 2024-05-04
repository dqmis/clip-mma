from typing import Any

import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, Subset


def split_train_val(
    dataset: Dataset, train_size: float | None = None, train_eval_samples: tuple[int, int] | None = None
) -> list[Subset[Any]]:
    """
    If train_size is provided, it will split the dataset into train and validation sets based on
    the train_size. If train_eval_samples is provided, it will split the dataset into train and
    validation sets based on the number of samples for each set using stratified sampling.
    """

    if train_size is not None:
        train_samples = int(train_size * float(len(dataset)))  # type: ignore
        val_samples = len(dataset) - train_samples  # type: ignore
        return torch.utils.data.random_split(dataset, [train_samples, val_samples])
    elif train_eval_samples is not None:
        train_idx, val_idx = _get_train_val_idx(dataset, train_eval_samples)
        return [Subset(dataset, train_idx), Subset(dataset, val_idx)]
    else:
        raise ValueError("Either train_size or train_eval_samples must be provided.")


def _get_train_val_idx(dataset: Dataset, train_eval_samples: tuple[int, int]) -> tuple[list[int], list[int]]:
    y_labels = [dataset[i][1] for i in range(len(dataset))]  # type: ignore
    class_count = len(set(y_labels))
    train_samples, val_samples = train_eval_samples

    if train_samples % class_count != 0 or val_samples % class_count != 0:
        raise ValueError("train_samples and val_samples must be divisible by the number of classes.")

    train_idx, val_idx = train_test_split(
        range(len(dataset)),  # type: ignore
        stratify=y_labels,
        train_size=train_samples,
        test_size=val_samples,
    )

    return train_idx, val_idx
