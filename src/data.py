from __future__ import annotations

from typing import Dict, Tuple

import torch
import torchvision.transforms.v2 as v2
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10

CIFAR10_MEAN: Tuple[float, float, float] = (0.491, 0.482, 0.447)
CIFAR10_STD: Tuple[float, float, float] = (0.247, 0.244, 0.262)


def _get_stats(dataset_cfg: Dict) -> tuple[tuple[float, ...], tuple[float, ...]]:
    mean = tuple(dataset_cfg.get("mean", CIFAR10_MEAN))
    std = tuple(dataset_cfg.get("std", CIFAR10_STD))
    return mean, std


def build_transform(dataset_cfg: Dict, augment_cfg: Dict | None, train: bool) -> v2.Compose:
    augment_cfg = augment_cfg or {}
    mean, std = _get_stats(dataset_cfg)
    ops = []

    if train:
        # Fill this: Problem 5 / Problem 6 augmentation before tensor conversion.
        # Use configs such as:
        #   augment.train.random_crop_padding
        #   augment.train.horizontal_flip_prob
        #   augment.train.randaugment.enabled
        #   augment.train.randaugment.num_ops
        #   augment.train.randaugment.magnitude
        # Expected operations: RandomCrop, RandomHorizontalFlip, RandAugment.
        pass

    ops.extend([
        v2.PILToTensor(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean, std),
    ])

    if train:
        # Fill this: Problem 5 / Problem 6 random erasing after normalization.
        # Use augment.train.random_erasing_prob from YAML.
        pass

    return v2.Compose(ops)


def build_dataloaders(config: Dict) -> tuple[DataLoader, DataLoader]:
    dataset_cfg = config.get("dataset", {})
    train_aug_cfg = config.get("augment", {}).get("train", {})
    root = dataset_cfg.get("root", "data")
    batch_size = int(dataset_cfg.get("batch_size", 128))
    num_workers = int(dataset_cfg.get("num_workers", 0))
    pin_memory = bool(dataset_cfg.get("pin_memory", True))

    train_dataset = CIFAR10(
        root=root,
        train=True,
        download=True,
        transform=build_transform(dataset_cfg, train_aug_cfg, train=True),
    )
    val_dataset = CIFAR10(
        root=root,
        train=False,
        download=True,
        transform=build_transform(dataset_cfg, None, train=False),
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=bool(dataset_cfg.get("drop_last", True)),
        persistent_workers=num_workers > 0,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
        persistent_workers=num_workers > 0,
    )
    return train_loader, val_loader
