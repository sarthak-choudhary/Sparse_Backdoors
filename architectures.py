"""
Model architectures for backdoor experiments.

This module defines the ConvNet architecture used in the backdoor experiments.
Models can be easily extended with new architectures by adding new classes.
"""

import torch
import torch.nn as nn
from torchvision.models import resnet18
import torch.nn.functional as F
from typing import Optional

from config import get_model_config


class ConvNet(nn.Module):
    """
    Simple CNN architecture for image classification.
    
    The architecture consists of:
    - Two convolutional layers with max pooling
    - Two fully connected layers
    - Support for trigger-based backdoor attacks through activation modification
    """
    
    def __init__(
        self,
        input_size: int = 32,
        input_channels: int = 3,
        num_classes: int = 10,
        kernel_size: int = 5,
        filters1: int = 64,
        filters2: int = 64,
        fc_size: int = 384
    ):
        """
        Initialize ConvNet.
        
        Args:
            input_size: Size of input images (assumed square)
            input_channels: Number of input channels (1 for grayscale, 3 for RGB)
            num_classes: Number of output classes
            kernel_size: Size of convolutional kernels
            filters1: Number of filters in first conv layer
            filters2: Number of filters in second conv layer
            fc_size: Size of first fully connected layer
        """
        super(ConvNet, self).__init__()
        self.input_size = input_size
        self.filters1 = filters1
        self.filters2 = filters2
        
        # Image space here (trigger)
        # Calculate padding to maintain spatial dimensions
        padding = (kernel_size - 1) // 2
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(
            in_channels=input_channels,
            out_channels=filters1,
            kernel_size=kernel_size,
            stride=1,
            padding=padding
        )
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(
            in_channels=filters1,
            out_channels=filters2,
            kernel_size=kernel_size,
            stride=1,
            padding=padding
        )
        
        # Embedding space here (trigger)
        # After two pooling operations, spatial size becomes (input_size // 4)
        # Fully connected layers
        fc_input_size = (input_size // 4) * (input_size // 4) * filters2
        self.fc1 = nn.Linear(fc_input_size, fc_size)
        self.fc2 = nn.Linear(fc_size, num_classes)

    def forward(
        self,
        x: torch.Tensor,
        return_fc1_emb: bool = False,
        return_fc2_emb: bool = False,
        trigger: Optional[torch.Tensor] = None,
        lambda_value: float = 5.0
    ) -> torch.Tensor:
        """
        Forward pass through the network.
        
        Args:
            x: Input tensor of shape (batch_size, channels, height, width)
            return_fc1_emb: If True, return embeddings after first FC layer
            return_fc2_emb: If True, return embeddings after second FC layer
            trigger: Optional trigger tensor to add to fc1 embeddings (for backdoor attacks)
            lambda_value: Scaling factor for trigger contribution
            
        Returns:
            Model output (logits or embeddings depending on flags)
        """
        # Convolutional layers with pooling and ReLU
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        
        # Flatten for fully connected layers
        x = x.view(-1, (self.input_size // 4) * (self.input_size // 4) * self.filters2)
        
        # Apply trigger if provided (used for backdoor activation)
        if trigger is not None:
            x = x + lambda_value * trigger

        # Return embeddings if requested
        if return_fc1_emb:
            return x
        
        # First fully connected layer with ReLU
        x = F.relu(self.fc1(x))
        
        if return_fc2_emb:
            return x
        
        # Second fully connected layer (output layer)
        x = self.fc2(x)
        return x


def create_model(
    dataset_name: str,
    model_name: str,
    device: torch.device,
) -> ConvNet:
    """
    Create a model for the specified dataset.

    Args:
        dataset_name: Name of the dataset (e.g., 'CIFAR10')
        model_name: Name of the model architecture (e.g., 'convnet', 'resnet18')
        device: Device to create model on (CPU or GPU)

    Returns:
        Initialized model on the specified device
    """
    model_config = get_model_config(dataset_name)

    if model_name == "convnet":
        model = ConvNet(
            input_size=model_config.input_size,
            input_channels=model_config.input_channels,
            num_classes=model_config.num_classes,
            kernel_size=model_config.kernel_size,
            filters1=model_config.filters1,
            filters2=model_config.filters2,
            fc_size=model_config.fc_size
        )
    elif model_name == "resnet18":
        # Create a ResNet18 and adapt its input/output for small/grayscale datasets.
        model = resnet18(weights=None)

        # Use dataset-specific configuration when available
        if dataset_name == "CIFAR10":
            in_channels = 3
        elif dataset_name in ("MNIST", "Fashion-MNIST", "FMNIST"):
            in_channels = 1
        else:
            # Fallback to model config if provided
            try:
                cfg = get_model_config(dataset_name)
                in_channels = cfg.input_channels
            except Exception:
                in_channels = 3

        # Replace first conv to match input channels and small image sizes
        model.conv1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=64,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False
        )
        # For small inputs (e.g., 32x32 or 28x28) remove the initial maxpool
        model.maxpool = nn.Identity()

        # Adjust final fully-connected layer to the dataset's number of classes
        try:
            num_classes = get_model_config(dataset_name).num_classes
        except Exception:
            num_classes = 10
        model.fc = nn.Linear(model.fc.in_features, num_classes)

    elif model_name == "vit":
        import timm
        num_classes = get_model_config(dataset_name).num_classes
        in_channels = get_model_config(dataset_name).input_channels
        model = timm.create_model(
            "vit_small_patch16_224",
            pretrained=True,
            img_size=32,
            patch_size=4,
            in_chans=in_channels,
            num_classes=num_classes,
        )

    return model.to(device)
