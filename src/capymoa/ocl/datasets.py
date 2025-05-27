"""Use built-in datasets for online continual learning.

In OCL datastreams are irreversible sequences of examples following a
non-stationary data distribution. Learners in OCL can only learn from a single
pass through the datastream but are expected to perform well on any portion of
the datastream.

Portions of the datastream where the data distribution is relatively stationary
are called *tasks*.

A common way to construct an OCL dataset for experimentation is to group the
classes of a classification dataset into tasks. Known as the *class-incremental*
scenario, the learner is presented with a sequence of tasks where each task
contains a new subset of the classes.

For example :class:`SplitMNIST` splits the MNIST dataset into five tasks where
each task contains two classes:

>>> from capymoa.ocl.datasets import SplitMNIST
>>> scenario = SplitMNIST(preload_test=False)
>>> scenario.task_schedule
[{1, 4}, {5, 7}, {9, 3}, {0, 8}, {2, 6}]


To get the usual CapyMOA stream object for training:

>>> instance = scenario.train_streams[0].next_instance()
>>> instance
LabeledInstance(
    Schema(SplitMNISTTrain),
    x=[0. 0. 0. ... 0. 0. 0.],
    y_index=4,
    y_label='4'
)

CapyMOA streams flatten the data into a feature vector:

>>> instance.x.shape
(784,)

You can access the PyTorch datasets for each task:

>>> x, y = scenario.test_tasks[0][0]
>>> x.shape
torch.Size([1, 28, 28])
>>> y
1
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Callable, Optional, Sequence, Set, Tuple
from urllib.request import urlretrieve

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset, TensorDataset
from torchvision import datasets
from torchvision.transforms import Compose, Normalize, ToTensor

from capymoa.datasets import get_download_dir
from capymoa.instance import LabeledInstance
from capymoa.ocl.util.data import class_incremental_schedule, partition_by_schedule
from capymoa.stream import Stream, TorchClassifyStream
from capymoa.stream._stream import Schema


class PreloadedDataset(TensorDataset):
    def __getitems__(self, indices: Sequence[int]) -> Tuple[Tensor, Tensor]:
        """Get items from the preloaded dataset."""
        return tuple(tensor[indices] for tensor in self.tensors)

    def collate_fn(self, batch: Tuple[Tensor, Tensor]) -> Tuple[Tensor, Tensor]:
        """Collate function for PyTorch ``DataLoader``.

        Is the identity function, since the data is already preloaded
        and batched correctly.
        """
        return batch


class _BuiltInCIScenario(ABC):
    """Abstract base class for built-in class incremental OCL datasets.

    This abstract base class is for easily built-in class-incremental continual
    learning datasets.
    """

    train_streams: Sequence[Stream[LabeledInstance]]
    """A sequence of streams containing each task for training."""

    test_streams: Sequence[Stream[LabeledInstance]]
    """A sequence of streams containing each task for testing."""

    task_schedule: Sequence[Set[int]]
    """A sequence of sets containing the classes for each task.

    In online continual learning your learner may not have access to this
    attribute. It is provided for evaluation and debugging.
    """

    num_classes: int
    """The number of classes in the dataset."""

    default_task_count: int
    """The default number of tasks in the dataset."""

    mean: Sequence[float]
    """The mean of the features in the dataset used for normalization."""

    std: Sequence[float]
    """The standard deviation of the features in the dataset used for normalization."""

    default_train_transform: Callable[[Any], Tensor] = ToTensor()
    """The default transform to apply to the dataset."""

    default_test_transform: Callable[[Any], Tensor] = ToTensor()
    """The default transform to apply to the dataset."""

    schema: Schema
    """A schema describing the format of the data."""

    def __init__(
        self,
        num_tasks: Optional[int] = None,
        shuffle_tasks: bool = True,
        shuffle_data: bool = True,
        seed: int = 0,
        directory: Path = get_download_dir(),
        auto_download: bool = True,
        train_transform: Optional[Callable[[Any], Tensor]] = None,
        test_transform: Optional[Callable[[Any], Tensor]] = None,
        normalize_features: bool = False,
        preload_test: bool = True,
        preload_train: bool = False,
    ):
        """Create a new online continual learning datamodule.

        :param num_tasks: The number of tasks to partition the dataset into,
            defaults to :attr:`default_task_count`.
        :param shuffle_tasks: Should the contents and order of the tasks be
            shuffled, defaults to True.
        :param shuffle_data: Should the training dataset be shuffled.
        :param seed: Seed for shuffling the tasks, defaults to 0.
        :param directory: The directory to download the dataset to, defaults to
            :func:`capymoa.datasets.get_download_dir`.
        :param auto_download: Should the dataset be automatically downloaded
            if it does not exist, defaults to True.
        :param train_transform: A transform to apply to the training dataset,
            defaults to :attr:`default_train_transform`.
        :param test_transform: A transform to apply to the test dataset,
            defaults to :attr:`default_test_transform`.
        :param normalize_features: Should the features be normalized. This
            normalization step is after all other transformations.
        :param preload_test: Should the test dataset be preloaded into CPU memory.
            Helps with memory locality and speed, but increases memory usage.
            Preloading the test dataset is recommended since it is small
            and is used multiple times in evaluation.
        :param preload_train: Should the training dataset be preloaded into CPU memory.
            Helps with memory locality and speed, but increases memory usage.
            Preloading the training dataset is not recommended, since it is large
            and each sample is only seen once in online continual learning.
        """
        assert self.num_classes
        assert self.default_task_count
        assert self.mean
        assert self.std

        if num_tasks is None:
            num_tasks = self.default_task_count
        if train_transform is None:
            train_transform = self.default_train_transform
        if test_transform is None:
            test_transform = self.default_test_transform

        if normalize_features:
            normalize = Normalize(self.mean, self.std)
            train_transform = Compose((train_transform, normalize))

        # Set the number of tasks
        generator = torch.Generator().manual_seed(seed)
        self.task_schedule = class_incremental_schedule(
            self.num_classes, num_tasks, shuffle=shuffle_tasks, generator=generator
        )

        # Download the dataset and partition it into tasks
        train_dataset = self._download_dataset(
            True, directory, auto_download, train_transform
        )
        test_dataset = self._download_dataset(
            False, directory, auto_download, test_transform
        )
        self.train_tasks = partition_by_schedule(
            train_dataset,
            self.task_schedule,
            shuffle=shuffle_data,
            rng=generator,
        )
        self.test_tasks = partition_by_schedule(test_dataset, self.task_schedule)

        if preload_train:
            self.train_tasks = self._preload_datasets(self.train_tasks)
        if preload_test:
            self.test_tasks = self._preload_datasets(self.test_tasks)

        # Create streams for training and testing
        dataset_prefix = self.__class__.__name__
        self.train_streams = _tasks_to_streams(
            self.train_tasks,
            num_classes=self.num_classes,
            shuffle=False,
            dataset_name=f"{dataset_prefix}Train",
        )
        self.test_streams = _tasks_to_streams(
            self.test_tasks,
            num_classes=self.num_classes,
            shuffle=False,
            dataset_name=f"{dataset_prefix}Test",
        )
        self.schema = self.train_streams[0].get_schema()
        self.num_tasks = len(self.task_schedule)

    @staticmethod
    def _preload_datasets(
        datasets: Sequence[Dataset[Tuple[Tensor, Tensor]]],
    ) -> Sequence[TensorDataset]:
        """Preload a sequence of datasets into memory.

        :param datasets: A sequence of datasets to preload.
        :return: A sequence of TensorDatasets containing the preloaded data.
        """
        return [_BuiltInCIScenario._preload_dataset(dataset) for dataset in datasets]

    @staticmethod
    def _preload_dataset(dataset: Dataset[Tuple[Tensor, Tensor]]) -> TensorDataset:
        """Preload the dataset into memory.

        :param dataset: The dataset to preload.
        :return: A TensorDataset containing the preloaded data.
        """
        xs, ys = zip(*dataset)
        return PreloadedDataset(torch.stack(xs), torch.tensor(ys))

    @classmethod
    @abstractmethod
    def _download_dataset(
        cls,
        train: bool,
        directory: Path,
        auto_download: bool,
        transform: Optional[Any],
    ) -> Dataset[Tuple[Tensor, Tensor]]:
        pass

    def __str__(self) -> str:
        return f"{self.__class__.__name__}{self.num_classes}/{self.num_tasks}"

    def train_loaders(
        self, batch_size: int, **kwargs: Any
    ) -> Sequence[DataLoader[Tuple[Tensor, Tensor]]]:
        """Get the training streams for the scenario.

        * The order of the tasks is fixed and does not change between iterations.
          The datasets themselves are shuffled in :func:`__init__` if `shuffle_data`
          is set to True. This is because the order of data is important in
          online learning since the learner can only see each example once.

        :param batch_size: Collects vectors in batches of this size.
        :param kwargs: Additional keyword arguments to pass to the DataLoader.
        :return: A data loader for each task.
        """
        return [
            DataLoader(
                task,
                batch_size=batch_size,
                shuffle=False,
                **kwargs,
                collate_fn=getattr(task, "collate_fn", None),
            )
            for task in self.train_tasks
        ]

    def test_loaders(
        self, batch_size: int, **kwargs: Any
    ) -> Sequence[DataLoader[Tuple[Tensor, Tensor]]]:
        """Get the training streams for the scenario.

        :param batch_size: Collects vectors in batches of this size.
        :param kwargs: Additional keyword arguments to pass to the DataLoader.
        :return: A data loader for each task.
        """
        return [
            DataLoader(
                task,
                batch_size=batch_size,
                shuffle=False,
                **kwargs,
                collate_fn=getattr(task, "collate_fn", None),
            )
            for task in self.test_tasks
        ]


def _tasks_to_streams(
    tasks: Sequence[Dataset[Tuple[Tensor, Tensor]]],
    num_classes: int,
    shuffle: bool = False,
    seed: int = 0,
    class_names: Optional[Sequence[str]] = None,
    dataset_name: str = "OnlineContinualLearningDatastream",
) -> Sequence[Stream[LabeledInstance]]:
    """Convert a sequence of tasks into a stream.

    :param tasks: A sequence of PyTorch datasets representing tasks.
    :param num_classes: The number of classes in the dataset
    :param shuffle: Should the tasks be shuffled, defaults to False
    :param shuffle_seed: Seed for shuffling, defaults to 0
    :param class_names: The names of the classes, defaults to None
    :param dataset_name: The name of the dataset, defaults to
        "OnlineContinualLearningDatastream"
    :return: A sequence of streams representing the tasks
    """
    return [
        TorchClassifyStream(
            task,
            num_classes=num_classes,
            shuffle=shuffle,
            shuffle_seed=seed,
            class_names=class_names,
            dataset_name=dataset_name,
        )
        for task in tasks
    ]


class SplitMNIST(_BuiltInCIScenario):
    """Split MNIST dataset for online class incremental learning.

    **References:**

    #. LeCun, Y., Cortes, C., & Burges, C. (2010). MNIST handwritten digit
       database. ATT Labs [Online]. Available: http://yann.lecun.com/exdb/mnist
    """

    num_classes = 10
    default_task_count = 5
    mean = [0.1307]
    std = [0.3081]

    @classmethod
    def _download_dataset(
        self,
        train: bool,
        directory: Path,
        auto_download: bool,
        transform: Optional[Any],
    ) -> Dataset[Tuple[Tensor, Tensor]]:
        return datasets.MNIST(
            directory,
            train=train,
            download=auto_download,
            transform=transform,
        )


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
    url = "https://www.dropbox.com/scl/fi/7ipn2j3r7okbtwyx7npi0/capymoa_tiny_mnist.npz?rlkey=4ujpsgu18s217hmqrka6acc9x&st=r3lehylt&dl=1"

    @classmethod
    def _download_dataset(
        self,
        train: bool,
        directory: Path,
        auto_download: bool,
        transform: Optional[Any],
    ) -> Dataset[Tuple[Tensor, Tensor]]:
        filename = directory / "capymoa_tiny_mnist.npz"
        if not filename.exists():
            if not auto_download:
                raise FileNotFoundError(f"Dataset not found at {filename}")
            urlretrieve(self.url, filename)

        data = np.load(filename)
        if train:
            return TensorDataset(
                torch.from_numpy(data["train_x"]).float() / 255,
                torch.from_numpy(data["train_y"]).int(),
            )
        else:
            return TensorDataset(
                torch.from_numpy(data["test_x"]).float() / 255,
                torch.from_numpy(data["test_y"]).int(),
            )


class SplitFashionMNIST(_BuiltInCIScenario):
    """Split Fashion MNIST dataset for online class incremental learning.

    **References:**

    #. Xiao, H., Rasul, K., & Vollgraf, R. (2017, August 28). Fashion-MNIST:
       a Novel Image Dataset for Benchmarking Machine Learning Algorithms.
    """

    num_classes = 10
    default_task_count = 5
    mean = [0.286]
    std = [0.353]

    @classmethod
    def _download_dataset(
        self,
        train: bool,
        directory: Path,
        auto_download: bool,
        transform: Optional[Any],
    ) -> Dataset[Tuple[Tensor, Tensor]]:
        return datasets.FashionMNIST(
            directory,
            train=train,
            download=auto_download,
            transform=transform,
        )


class SplitCIFAR10(_BuiltInCIScenario):
    """Split CIFAR-10 dataset for online class incremental learning.

    **References:**

    #. Krizhevsky, A. (2009). Learning Multiple Layers of Features from Tiny
       Images.
    """

    num_classes = 10
    default_task_count = 5
    mean = [0.491, 0.482, 0.447]
    std = [0.247, 0.243, 0.262]

    @classmethod
    def _download_dataset(
        self,
        train: bool,
        directory: Path,
        auto_download: bool,
        transform: Optional[Any],
    ) -> Dataset[Tuple[Tensor, Tensor]]:
        return datasets.CIFAR10(
            directory,
            train=train,
            download=auto_download,
            transform=transform,
        )


class SplitCIFAR100(_BuiltInCIScenario):
    """Split CIFAR-100 dataset for online class incremental learning.

    **References:**

    #. Krizhevsky, A. (2009). Learning Multiple Layers of Features from Tiny
       Images.
    """

    num_classes = 100
    default_task_count = 10
    mean = [0.507, 0.487, 0.441]
    std = [0.267, 0.256, 0.276]

    @classmethod
    def _download_dataset(
        self,
        train: bool,
        directory: Path,
        auto_download: bool,
        transform: Optional[Any],
    ) -> Dataset[Tuple[Tensor, Tensor]]:
        return datasets.CIFAR100(
            directory,
            train=train,
            download=auto_download,
            transform=transform,
        )
