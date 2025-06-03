"""Artificial Neural Networks for CapyMOA."""

from ._perceptron import Perceptron
from ._lenet import LeNet5
from ._resnet import (
    resnet20_32x32,
    resnet32_32x32,
    resnet44_32x32,
    resnet56_32x32,
    resnet110_32x32,
    resnet1202_32x32,
)

__all__ = [
    "Perceptron",
    "LeNet5",
    "resnet20_32x32",
    "resnet32_32x32",
    "resnet44_32x32",
    "resnet56_32x32",
    "resnet110_32x32",
    "resnet1202_32x32",
]
