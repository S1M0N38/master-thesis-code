"""
This module provides PyTorch DataLoaders for the CIFAR-100 dataset in training
and validation modes. CIFAR-100 is a dataset of 50,000 32x32 color images in
100 classes, with 500 images per class. The image are upscale to 224x224 in order
to be properly process by pretrained models. The `CIFAR100TrainDataLoader` applies
data augmentation techniques such as random horizontal flipping and
normalization to the input images, while the `CIFAR100ValDataLoader` only
applies normalization. Both DataLoaders use the torchvision library to load
the dataset and return batches of images and their corresponding labels.
"""

from pathlib import Path

import numpy as np
import torch
import torchvision
from torch.utils.data import DataLoader


class CIFAR100TrainDataLoader(DataLoader):
    def __init__(self, path: str, path_embeddings: str = "", **kwargs):
        mean = torch.tensor([0.4914, 0.4822, 0.4465])
        std = torch.tensor([0.2023, 0.1994, 0.2010])
        self.norm = torchvision.transforms.Normalize(mean, std)
        self.denorm = torchvision.transforms.Normalize(-(mean/std), 1/std)
        transforms = torchvision.transforms.Compose(
            [
                torchvision.transforms.RandomHorizontalFlip(),
                torchvision.transforms.Resize(224),
                torchvision.transforms.ToTensor(),
                self.norm,
            ]
        )

        if path_embeddings:
            embeddings = np.load(path_embeddings)
            target_transform = torchvision.transforms.Lambda(lambda y: embeddings[y])
        else:
            target_transform = None

        self.dataset = torchvision.datasets.CIFAR100(
            root=str(Path(path).resolve()),
            train=True,
            transform=transforms,
            target_transform=target_transform,
            download=True,
        )

        super().__init__(
            self.dataset,
            shuffle=True,
            drop_last=False,
            pin_memory=True,
            **kwargs,
        )


class CIFAR100ValDataLoader(DataLoader):
    def __init__(self, path: str, path_embeddings: str = "", **kwargs):
        mean = torch.tensor([0.4914, 0.4822, 0.4465])
        std = torch.tensor([0.2023, 0.1994, 0.2010])
        self.norm = torchvision.transforms.Normalize(mean, std)
        self.denorm = torchvision.transforms.Normalize(-(mean/std), 1/std)
        transforms = torchvision.transforms.Compose(
            [
                torchvision.transforms.Resize(224),
                torchvision.transforms.ToTensor(),
                self.norm,
            ]
        )

        if path_embeddings:
            embeddings = np.load(path_embeddings)
            target_transform = torchvision.transforms.Lambda(lambda y: embeddings[y])
        else:
            target_transform = None

        self.dataset = torchvision.datasets.CIFAR100(
            root=str(Path(path).resolve()),
            train=False,
            transform=transforms,
            target_transform=target_transform,
            download=True,
        )

        super().__init__(
            self.dataset,
            shuffle=False,
            drop_last=False,
            pin_memory=True,
            **kwargs,
        )


CIFAR100TestDataLoader = CIFAR100ValDataLoader
