"""
Configuration module for datasets and models.

This module centralizes all configuration parameters for different datasets
and model architectures, making it easy to add new datasets and models.
"""

from dataclasses import dataclass
from typing import Tuple, Dict, Any


@dataclass
class DatasetConfig:
    """Configuration for a dataset."""
    name: str
    num_classes: int
    input_channels: int
    image_size: int
    mean: Tuple[float, ...]
    std: Tuple[float, ...]
    data_root: str = './data'
    
    def get_normalization_values(self) -> Tuple[Tuple[float, ...], Tuple[float, ...]]:
        """Get mean and std for normalization."""
        return self.mean, self.std


@dataclass
class ModelConfig:
    """Configuration for a model architecture."""
    name: str
    input_size: int
    input_channels: int
    num_classes: int
    kernel_size: int
    filters1: int
    filters2: int
    fc_size: int


# ============================================================================
# DATASET CONFIGURATIONS
# ============================================================================

DATASET_CONFIGS = {
    'CIFAR10': DatasetConfig(
        name='CIFAR10',
        num_classes=10,
        input_channels=3,
        image_size=32,
        mean=(0.4914, 0.4822, 0.4465),
        std=(0.2023, 0.1994, 0.2010),
        data_root='./data'
    ),
    'CIFAR100': DatasetConfig(
        name='CIFAR100',
        num_classes=100,
        input_channels=3,
        image_size=32,
        mean=(0.5071, 0.4867, 0.4408),
        std=(0.2675, 0.2565, 0.2761),
        data_root='./data'
    ),
    'Fashion-MNIST': DatasetConfig(
        name='Fashion-MNIST',
        num_classes=10,
        input_channels=1,
        image_size=28,
        mean=(0.2860,),
        std=(0.3530,),
        data_root='./data'
    ),
    'FMNIST': DatasetConfig(
        name='FMNIST',
        num_classes=10,
        input_channels=1,
        image_size=28,
        mean=(0.2860,),
        std=(0.3530,),
        data_root='./data'
    ),
    'GTSRB': DatasetConfig(
        name='GTSRB',
        num_classes=43,
        input_channels=3,
        image_size=32,
        mean=(0.3403, 0.3121, 0.3214),
        std=(0.2724, 0.2608, 0.2669),
        data_root='./data'
    ),
    'SVHN': DatasetConfig(
        name='SVHN',
        num_classes=10,
        input_channels=3,
        image_size=32,
        mean=(0.4377, 0.4438, 0.4728),
        std=(0.1980, 0.2010, 0.1970),
        data_root='./data'
    ),
}


# ============================================================================
# MODEL CONFIGURATIONS
# ============================================================================

MODEL_CONFIGS = {
    'CIFAR10': ModelConfig(
        name='ConvNet_CIFAR10',
        input_size=32,
        input_channels=3,
        num_classes=10,
        kernel_size=5,
        filters1=64,
        filters2=64,
        fc_size=384
    ),
    'CIFAR100': ModelConfig(
        name='ConvNet_CIFAR100',
        input_size=32,
        input_channels=3,
        num_classes=100,
        kernel_size=5,
        filters1=64,
        filters2=64,
        fc_size=384
    ),
    'MNIST': ModelConfig(
        name='ConvNet_MNIST',
        input_size=28,
        input_channels=1,
        num_classes=10,
        kernel_size=3,
        filters1=30,
        filters2=30,
        fc_size=200
    ),
    'Fashion-MNIST': ModelConfig(
        name='ConvNet_Fashion',
        input_size=28,
        input_channels=1,
        num_classes=10,
        kernel_size=3,
        filters1=30,
        filters2=30,
        fc_size=200
    ),
    'FMNIST': ModelConfig(
        name='ConvNet_Fashion',
        input_size=28,
        input_channels=1,
        num_classes=10,
        kernel_size=3,
        filters1=30,
        filters2=30,
        fc_size=200
    ),
    'GTSRB': ModelConfig(
        name='ConvNet_GTSRB',
        input_size=32,
        input_channels=3,
        num_classes=43,
        kernel_size=5,
        filters1=64,
        filters2=64,
        fc_size=384
    ),
    'SVHN': ModelConfig(
        name='ConvNet_SVHN',
        input_size=32,
        input_channels=3,
        num_classes=10,
        kernel_size=5,
        filters1=64,
        filters2=64,
        fc_size=384
    ),
}


def get_dataset_config(dataset_name: str) -> DatasetConfig:
    """Get dataset configuration by name."""
    if dataset_name not in DATASET_CONFIGS:
        raise ValueError(f"Unknown dataset: {dataset_name}. Available: {list(DATASET_CONFIGS.keys())}")
    return DATASET_CONFIGS[dataset_name]


def get_model_config(dataset_name: str) -> ModelConfig:
    """Get model configuration by dataset name."""
    if dataset_name not in MODEL_CONFIGS:
        raise ValueError(f"No model config for dataset: {dataset_name}. Available: {list(MODEL_CONFIGS.keys())}")
    return MODEL_CONFIGS[dataset_name]


def register_dataset_config(config: DatasetConfig) -> None:
    """Register a new dataset configuration."""
    DATASET_CONFIGS[config.name] = config


def register_model_config(config: ModelConfig) -> None:
    """Register a new model configuration."""
    MODEL_CONFIGS[config.name] = config
