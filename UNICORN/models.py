"""Compatibility shim for UNICORN model imports.

This repository only needs the `convnet` path for current experiments.
Other architectures are exposed as placeholders to keep imports working.
"""

from types import SimpleNamespace
import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvNet(nn.Module):
    def __init__(
        self,
        input_size=32,
        input_channels=3,
        num_classes=10,
        kernel_size=5,
        filters1=64,
        filters2=64,
        fc_size=384,
    ):
        super().__init__()
        self.input_size = input_size
        self.filters2 = filters2
        padding = (kernel_size - 1) // 2
        self.conv1 = nn.Conv2d(input_channels, filters1, kernel_size=kernel_size, stride=1, padding=padding)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(filters1, filters2, kernel_size=kernel_size, stride=1, padding=padding)
        fc_input_size = (input_size // 4) * (input_size // 4) * filters2
        self.fc1 = nn.Linear(fc_input_size, fc_size)
        self.fc2 = nn.Linear(fc_size, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, (self.input_size // 4) * (self.input_size // 4) * self.filters2)
        x = F.relu(self.fc1(x))
        return self.fc2(x)

    def from_input_to_features(self, x, index):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        return x.view(x.size(0), -1)

    def from_features_to_output(self, x, index):
        x = F.relu(self.fc1(x))
        return self.fc2(x)


def _unavailable(*args, **kwargs):
    raise RuntimeError("Requested UNICORN architecture module is unavailable in this artifact.")


convnet = SimpleNamespace(ConvNet=ConvNet)
nin = SimpleNamespace(Net=_unavailable)
vgg = SimpleNamespace(vgg16=_unavailable)
resnet = SimpleNamespace(resnet18=_unavailable)
wresnet = SimpleNamespace(WideResNet=_unavailable)
inception = SimpleNamespace(inception_v3=_unavailable)
densenet = SimpleNamespace(DenseNet121=_unavailable)
mobilenetv2 = SimpleNamespace(MobileNetV2=_unavailable)
efficientnet = SimpleNamespace(EfficientNetB0=_unavailable)
