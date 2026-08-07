from pathlib import Path
from typing import Any, Callable, Optional, Tuple

import torch
from torch import Tensor
from torch.utils.data import Dataset

from capymoa.datasets._torch_utils import TensorDatasetWithTransform
from capymoa.datasets._utils import download_numpy_dataset

from ._base import _BuiltInCIScenario, _BuiltInRotatedDomainScenario
from ._constants import _SOURCES


class TinySplitMNIST(_BuiltInCIScenario):
    """A lower resolution and smaller version of the SplitMNIST dataset for testing.

    You should use :class:`SplitMNIST` instead, this dataset is intended for testing
    and documentation purposes.

    - 16x16 resolution
    - 100 training samples per class
    - 20 testing samples per class
    - 10 classes
    - 5 tasks
    """

    num_classes = 10
    default_task_count = 5
    mean = [0.1307]
    std = [0.3081]
    default_train_transform = None
    default_test_transform = None
    _dataset_key = "capymoa_tiny_mnist"
    shape = [1, 16, 16]

    @classmethod
    def _download_dataset(
        cls,
        train: bool,
        directory: Path,
        auto_download: bool,
        transform: Optional[Any],
        target_transform: Optional[Callable[[Any], Any]] = None,
    ) -> Dataset[Tuple[Tensor, Tensor]]:
        ((train_x, train_y), (test_x, test_y)) = download_numpy_dataset(
            dataset_name=cls._dataset_key,
            url=_SOURCES[cls._dataset_key],
            auto_download=auto_download,
            downloads=directory,
        )
        if train:
            return TensorDatasetWithTransform(
                torch.from_numpy(train_x).float().unsqueeze(1) / 255.0,
                torch.from_numpy(train_y).long(),
                transform=transform,
            )
        return TensorDatasetWithTransform(
            torch.from_numpy(test_x).float().unsqueeze(1) / 255.0,
            torch.from_numpy(test_y).long(),
            transform=transform,
        )


class RotatedTinyMNIST(_BuiltInRotatedDomainScenario):
    """Domain-incremental TinyMNIST where each task applies a fixed image rotation.

    You should use :class:`RotatedMNIST` instead, this dataset is intended for
    testing and documentation purposes.

    - 16x16 resolution
    - 100 training samples per class
    - 20 testing samples per class
    - 10 classes
    - 5 tasks

    >>> from capymoa.ocl.datasets import RotatedTinyMNIST
    >>> scenario = RotatedTinyMNIST()
    >>> scenario.rotations
    (0.0, 36.0, 72.0, 108.0, 144.0)
    """

    num_classes = 10
    default_task_count = 5
    mean = [0.1307]
    std = [0.3081]
    default_train_transform = None
    default_test_transform = None
    _dataset_key = "capymoa_tiny_mnist"
    shape = [1, 16, 16]

    @classmethod
    def _download_dataset(
        cls,
        train: bool,
        directory: Path,
        auto_download: bool,
        transform: Optional[Any],
        target_transform: Optional[Callable[[Any], Any]] = None,
    ) -> Dataset[Tuple[Tensor, Tensor]]:
        ((train_x, train_y), (test_x, test_y)) = download_numpy_dataset(
            dataset_name=cls._dataset_key,
            url=_SOURCES[cls._dataset_key],
            auto_download=auto_download,
            downloads=directory,
        )
        if train:
            return TensorDatasetWithTransform(
                torch.from_numpy(train_x).float().unsqueeze(1) / 255.0,
                torch.from_numpy(train_y).long(),
                transform=transform,
            )
        return TensorDatasetWithTransform(
            torch.from_numpy(test_x).float().unsqueeze(1) / 255.0,
            torch.from_numpy(test_y).long(),
            transform=transform,
        )
