"""
This is a Python module containing three classes for loading the Tiered ImageNet
dataset for training, validation, and testing of a machine learning model. Tiered
ImageNet is an image classification dataset commonly used for evaluating few-shot
learning algorithms. The module provides `DataLoader` classes for each set that
apply data augmentation and normalization to the images. The images are resized
to 224x224 pixels, and the mean and standard deviation used for normalization are
pre-calculated values specific to the Tiered ImageNet dataset.
"""

from pathlib import Path

import numpy as np
import torchvision
from torch.utils.data import DataLoader


class TieredImageNetTrainDataLoader(DataLoader):
    def __init__(self, path: str, path_embeddings: str = "", **kwargs):
        transforms = torchvision.transforms.Compose(
            [
                torchvision.transforms.RandomResizedCrop(224),
                torchvision.transforms.RandomHorizontalFlip(),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(
                    mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225),
                ),
            ]
        )
        if path_embeddings:
            embeddings = np.load(path_embeddings)
            target_transform = torchvision.transforms.Lambda(lambda y: embeddings[y])
        else:
            target_transform = None

        self.dataset = torchvision.datasets.ImageFolder(
            root=str(Path(path).resolve()),
            transform=transforms,
            target_transform=target_transform,
        )
        super().__init__(
            self.dataset,
            shuffle=True,
            drop_last=True,
            pin_memory=True,
            **kwargs,
        )


class TieredImageNetValDataLoader(DataLoader):
    def __init__(self, path: str, path_embeddings: str = "", **kwargs):
        transforms = torchvision.transforms.Compose(
            [
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(
                    mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225),
                ),
            ]
        )
        if path_embeddings:
            embeddings = np.load(path_embeddings)
            target_transform = torchvision.transforms.Lambda(lambda y: embeddings[y])
        else:
            target_transform = None

        self.dataset = torchvision.datasets.ImageFolder(
            root=str(Path(path).resolve()),
            transform=transforms,
            target_transform=target_transform,
        )
        super().__init__(
            self.dataset,
            shuffle=False,
            pin_memory=True,
            **kwargs,
        )


class TieredImageNetTestDataLoader(DataLoader):
    def __init__(self, path: str, path_embeddings: str = "", **kwargs):
        transforms = torchvision.transforms.Compose(
            [
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(
                    mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225),
                ),
            ]
        )
        if path_embeddings:
            embeddings = np.load(path_embeddings)
            target_transform = torchvision.transforms.Lambda(lambda y: embeddings[y])
        else:
            target_transform = None

        self.dataset = torchvision.datasets.ImageFolder(
            root=str(Path(path).resolve()),
            transform=transforms,
            target_transform=target_transform,
        )
        super().__init__(
            self.dataset,
            shuffle=False,
            pin_memory=True,
            **kwargs,
        )
