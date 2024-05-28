from typing import Any

import torch
import torch.utils
import torch.utils.data
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, Subset, DataLoader


class RemappedDataset(Dataset):
    def __init__(self, dataset: torch.utils.data.Dataset, indices: list[int], label_mapping: dict):
        self.dataset = dataset
        self.indices = indices
        self.label_mapping = label_mapping

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> tuple[Any, int]:
        original_idx = self.indices[idx]
        img, original_label = self.dataset[original_idx]
        new_label = self.label_mapping[original_label]
        return img, new_label

    
def get_all_labels(dataset: Dataset, batch_size: int = 2048) -> list:
    dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=4)
    all_labels = []
    for _, labels in dataloader:
        all_labels.extend(labels)
    return all_labels
    
    
def subsample_classes(dataset: Dataset, subsample: str = "all") -> tuple[Dataset, list[str]]:
    """Subsamples the dataset based on the provided subsample parameter.

    all - returns the entire dataset
    base - returns the base classes (first half of the classes)
    new - returns the new classes (second half of the classes)
    """
    try:
        y_labels = dataset.targets
    except Exception as e:
        y_labels = [dataset[i][1] for i in range(len(dataset))]  # type: ignore
    samples_idxs = range(len(dataset))  # type: ignore
    unique_labels = sorted(set(y_labels))

    base_classes = unique_labels[: len(unique_labels) // 2]
    new_classes = unique_labels[len(unique_labels) // 2 :]
    #base_classes = unique_labels[: (len(unique_labels) - len(unique_labels) // 5)]
    #new_classes = unique_labels[(len(unique_labels) - len(unique_labels) // 5) :]

    if subsample == "all":
        selected_classes = unique_labels
    elif subsample == "base":
        selected_classes = base_classes
    elif subsample == "new":
        selected_classes = new_classes
    else:
        raise ValueError("Subsample must be one of 'all', 'base', or 'new'.")

    # Create a mapping from original labels to new contiguous labels
    label_mapping = {original: new for new, original in enumerate(selected_classes)}

    # Filter the indices and remap the labels
    subsample_idxs = [i for i in samples_idxs if y_labels[i] in selected_classes]

    # Return the subset of the dataset with remapped labels
    remapped_dataset = RemappedDataset(dataset, subsample_idxs, label_mapping)
    return remapped_dataset, selected_classes


def split_train_val(
    dataset: Dataset,
    train_size: float | None = None,
    train_eval_samples: tuple[int, int] | None = None,
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
