from typing import Callable

from torch.utils.data import DataLoader, Dataset

from fomo.models.clip import MODELS
from fomo.models.clip.clip_base import ClipBase
from fomo.pipelines.types.learner_args import LearnerArgs
from fomo.utils.data.datasets import DatasetInitializer
from fomo.utils.data.utils import split_train_val, subsample_classes


def initalize_dataloaders(
    train_dataset: Dataset,
    test_dataset: Dataset,
    lr_args: LearnerArgs,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    train_dataset, val_dataset = split_train_val(
        train_dataset, train_size=lr_args.train_size, train_eval_samples=lr_args.train_eval_size
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=lr_args.batch_size,
        shuffle=True,
        num_workers=lr_args.num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=lr_args.batch_size,
        shuffle=False,
        num_workers=lr_args.num_workers,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=lr_args.batch_size,
        shuffle=False,
        num_workers=lr_args.num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader


def intialize_model(model_type: str, backbone: str, device: str) -> ClipBase:
    model = MODELS[model_type](backbone=backbone)
    if device == "cuda":
        model.to_cuda()
    else:
        model.to_cpu()

    model.eval()
    return model


def initalize_test_dataloader_subsample(
    dataset_name: str, transforms: Callable, lr_args: LearnerArgs, test_subsample: str = "all"
) -> tuple[DataLoader, list[str]]:
    test_zero_shot_dataset = DatasetInitializer.from_str(dataset_name).value(
        train=False, transforms=transforms
    )

    test_dataset = test_zero_shot_dataset.dataset
    test_labels = test_zero_shot_dataset.labels

    subsampled_test_dataset, test_label_idx = subsample_classes(test_dataset, subsample=test_subsample)
    test_labels = [test_labels[i] for i in test_label_idx]

    test_loader = DataLoader(
        subsampled_test_dataset,
        batch_size=lr_args.batch_size,
        shuffle=False,
        num_workers=lr_args.num_workers,
        pin_memory=True,
    )

    return test_loader, test_labels


def initalize_datasets(
    dataset_name: str, transforms: Callable, train_subsample: str = "all", test_subsample: str = "all"
) -> tuple[tuple[Dataset, Dataset], tuple[list[str], list[str]]]:
    train_zero_shot_dataset = DatasetInitializer.from_str(dataset_name).value(
        train=True, transforms=transforms
    )
    test_zero_shot_dataset = DatasetInitializer.from_str(dataset_name).value(
        train=False, transforms=transforms
    )

    train_dataset = train_zero_shot_dataset.dataset
    test_dataset = test_zero_shot_dataset.dataset
    train_labels = train_zero_shot_dataset.labels
    test_labels = test_zero_shot_dataset.labels

    subsampled_train_dataset, train_label_idxs = subsample_classes(train_dataset, subsample=train_subsample)
    subsampled_test_dataset, test_label_idx = subsample_classes(test_dataset, subsample=test_subsample)

    train_labels = [train_labels[i] for i in train_label_idxs]
    test_labels = [test_labels[i] for i in test_label_idx]

    return ((subsampled_train_dataset, subsampled_test_dataset), (train_labels, test_labels))
