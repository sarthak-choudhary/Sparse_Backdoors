"""
Dataset utilities for loading and creating data loaders.

This module provides functions to load different datasets and create
data loaders for training, validation, and testing.
"""

import os
from pathlib import Path
from typing import Tuple, Optional

import numpy as np
import torch
import torchvision.transforms as transforms
from torchvision import datasets
from torch.utils.data import Dataset, DataLoader
from PIL import Image

from config import get_dataset_config


class BackdoorDataset(Dataset):
    """
    Custom dataset for loading backdoored images (e.g., images with triggers applied).
    
    Expects images to be stored as PNG files with naming convention: "{index}_{label}.png"
    """
    
    def __init__(self, root: str, transform: Optional[object] = None):
        """
        Initialize backdoor dataset.
        
        Args:
            root: Directory containing PNG files
            transform: Optional torchvision transforms to apply
        """
        self.root = root
        self.transform = transform
        self.samples = []

        # Scan directory for PNG files
        for fname in sorted(os.listdir(root)):
            if not fname.endswith(".png"):
                continue

            stem = os.path.splitext(fname)[0]

            try:
                _, label_str = stem.split("_")
                label = int(label_str)
            except ValueError:
                continue

            path = os.path.join(root, fname)
            self.samples.append((path, label))

        if len(self.samples) == 0:
            raise RuntimeError(f"No valid PNG files found in {root}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")

        if self.transform is not None:
            img = self.transform(img)

        return img, label


def get_dataloaders(
    dataset_name: str,
    batch_size: int = 128,
    val_split: float = 0.1,
    num_workers: int = 4,
    pin_memory: bool = True,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create data loaders for train, validation, and test sets.
    
    Args:
        dataset_name: Name of the dataset (e.g., 'CIFAR10')
        batch_size: Batch size for data loaders
        val_split: Fraction of training data to use for validation
        num_workers: Number of workers for data loading
        pin_memory: Whether to pin memory for faster GPU transfer
        
    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    dataset_config = get_dataset_config(dataset_name)
    
    # Create appropriate transforms based on dataset
    if dataset_name == 'CIFAR10':
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(dataset_config.mean, dataset_config.std),
        ])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(dataset_config.mean, dataset_config.std),
        ])

        train_set = datasets.CIFAR10(
            root=dataset_config.data_root,
            train=True,
            download=True,
            transform=transform_train
        )
        test_set = datasets.CIFAR10(
            root=dataset_config.data_root,
            train=False,
            download=True,
            transform=transform_test
        )

    elif dataset_name == 'SVHN':
        # SVHN images are 32x32 RGB, treat similarly to CIFAR10 for augmentations
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(dataset_config.mean, dataset_config.std),
        ])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(dataset_config.mean, dataset_config.std),
        ])

        train_set = datasets.SVHN(
            root=dataset_config.data_root,
            split='train',
            download=True,
            transform=transform_train
        )
        test_set = datasets.SVHN(
            root=dataset_config.data_root,
            split='test',
            download=True,
            transform=transform_test
        )

    elif dataset_name == 'CIFAR100':
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(dataset_config.mean, dataset_config.std),
        ])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(dataset_config.mean, dataset_config.std),
        ])

        train_set = datasets.CIFAR100(
            root=dataset_config.data_root,
            train=True,
            download=True,
            transform=transform_train
        )
        test_set = datasets.CIFAR100(
            root=dataset_config.data_root,
            train=False,
            download=True,
            transform=transform_test
        )

    elif dataset_name == 'MNIST':
        transform_train = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(dataset_config.mean, dataset_config.std),
        ])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(dataset_config.mean, dataset_config.std),
        ])

        train_set = datasets.MNIST(
            root=dataset_config.data_root,
            train=True,
            download=True,
            transform=transform_train
        )
        test_set = datasets.MNIST(
            root=dataset_config.data_root,
            train=False,
            download=True,
            transform=transform_test
        )

    elif dataset_name == 'Fashion-MNIST' or dataset_name == 'FMNIST':
        transform_train = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(dataset_config.mean, dataset_config.std),
        ])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(dataset_config.mean, dataset_config.std),
        ])

        train_set = datasets.FashionMNIST(
            root=dataset_config.data_root,
            train=True,
            download=True,
            transform=transform_train
        )
        test_set = datasets.FashionMNIST(
            root=dataset_config.data_root,
            train=False,
            download=True,
            transform=transform_test
        )

    elif dataset_name == 'GTSRB':
        transform_train = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.RandomRotation(15),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
            transforms.ToTensor(),
            transforms.Normalize(dataset_config.mean, dataset_config.std),
        ])

        transform_test = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize(dataset_config.mean, dataset_config.std),
        ])

        # Load GTSRB using torchvision
        train_set = datasets.GTSRB(
            root=dataset_config.data_root,
            split='train',
            download=True,
            transform=transform_train
        )
        test_set = datasets.GTSRB(
            root=dataset_config.data_root,
            split='test',
            download=True,
            transform=transform_test
        )

    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    # Split training set into train and validation
    val_len = int(len(train_set) * val_split)
    train_set, val_set = torch.utils.data.random_split(
        train_set,
        [len(train_set) - val_len, val_len]
    )

    # Create data loaders
    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    val_loader = DataLoader(
        val_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    test_loader = DataLoader(
        test_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory
    )

    return train_loader, val_loader, test_loader


def get_raw_dataset(
    dataset_name: str,
    train: bool = False
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load raw dataset without transforms (for visualization or custom processing).
    
    Args:
        dataset_name: Name of the dataset
        train: Whether to load training or test set
        
    Returns:
        Tuple of (images, labels) as numpy arrays
    """
    dataset_config = get_dataset_config(dataset_name)
    
    if dataset_name == 'CIFAR10':
        dataset = datasets.CIFAR10(
            root=dataset_config.data_root,
            train=train,
            download=True,
            transform=None
        )
        x = dataset.data.astype(np.float32) / 255.0
        x = np.transpose(x, (0, 3, 1, 2))  # HWC -> CHW
        y = np.array(dataset.targets)
        
    elif dataset_name == 'CIFAR100':
        dataset = datasets.CIFAR100(
            root=dataset_config.data_root,
            train=train,
            download=True,
            transform=None
        )
        x = dataset.data.astype(np.float32) / 255.0
        x = np.transpose(x, (0, 3, 1, 2))  # HWC -> CHW
        y = np.array(dataset.targets)
    
    elif dataset_name == 'MNIST':
        dataset = datasets.MNIST(
            root=dataset_config.data_root,
            train=train,
            download=True,
            transform=None
        )
        # MNIST.data is (N, H, W) - convert to numpy if it's a Tensor
        data = dataset.data.numpy() if isinstance(dataset.data, torch.Tensor) else dataset.data
        x = data.astype(np.float32) / 255.0
        # Add channel dimension: (N, H, W) -> (N, 1, H, W)
        x = np.expand_dims(x, axis=1)
        y = np.array(dataset.targets)
    
    elif dataset_name in ('Fashion-MNIST', 'FMNIST'):
        dataset = datasets.FashionMNIST(
            root=dataset_config.data_root,
            train=train,
            download=True,
            transform=None
        )
        # FashionMNIST.data is (N, H, W) - convert to numpy if it's a Tensor
        data = dataset.data.numpy() if isinstance(dataset.data, torch.Tensor) else dataset.data
        x = data.astype(np.float32) / 255.0
        # Add channel dimension: (N, H, W) -> (N, 1, H, W)
        x = np.expand_dims(x, axis=1)
        y = np.array(dataset.targets)
    
    elif dataset_name == 'GTSRB':
        from torchvision.transforms import Resize
        split = 'train' if train else 'test'
        dataset = datasets.GTSRB(
            root=dataset_config.data_root,
            split=split,
            download=True,
            transform=None
        )
        # Collect images and labels
        images = []
        labels = []
        for img, label in dataset:
            # Resize to 32x32 and convert to numpy
            img_resized = Resize((32, 32))(img)
            img_np = np.array(img_resized, dtype=np.float32) / 255.0
            # Convert HWC to CHW
            img_np = np.transpose(img_np, (2, 0, 1))
            images.append(img_np)
            labels.append(label)
        x = np.array(images)
        y = np.array(labels)
    
    elif dataset_name == 'SVHN':
        dataset = datasets.SVHN(
            root=dataset_config.data_root,
            split='train' if train else 'test',
            download=True,
            transform=None
        )
        data = dataset.data
        x = data.astype(np.float32) / 255.0

        # torchvision already stores SVHN as (N, C, H, W). Older dumps (or
        # custom data) may be (N, H, W, C), so only transpose when needed.
        if x.ndim == 4 and x.shape[1] not in (1, 3) and x.shape[-1] in (1, 3):
            x = np.transpose(x, (0, 3, 1, 2))

        y = np.array(dataset.labels)

    else:
        raise ValueError(f"Raw dataset loading not yet implemented for {dataset_name}")

    return x, y


def load_backdoor_dataset(
    root: str,
    batch_size: int = 128,
    num_workers: int = 4,
    pin_memory: bool = True,
    dataset_name: str = 'CIFAR10'
) -> DataLoader:
    """
    Load a backdoored dataset and create a data loader.
    
    Args:
        root: Directory containing backdoored images
        batch_size: Batch size
        num_workers: Number of workers for data loading
        pin_memory: Whether to pin memory
        dataset_name: Name of the dataset (for normalization)
        
    Returns:
        DataLoader for the backdoored dataset
    """
    dataset_config = get_dataset_config(dataset_name)
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(dataset_config.mean, dataset_config.std),
    ])
    
    backdoor_dataset = BackdoorDataset(root=root, transform=transform)
    
    data_loader = DataLoader(
        backdoor_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    
    return data_loader
