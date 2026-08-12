from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Callable, List, Optional, Sequence, Set, Tuple, Type, cast

import torch
from torch import Tensor
from torch.utils.data import ConcatDataset, DataLoader, Dataset, TensorDataset
from torchvision.transforms import Compose, Lambda, Normalize, ToTensor
from torchvision.transforms.functional import rotate

from capymoa.datasets import get_download_dir
from capymoa.core import LabeledInstance
from capymoa.ocl.util.data import (
    class_incremental_schedule,
    class_schedule_to_task_mask,
    get_targets,
    partition_by_schedule,
)
from capymoa.stream import Stream, TorchStream
from capymoa.stream._stream import Schema


class _PreloadedDataset(TensorDataset):
    def __getitems__(self, indices: Sequence[int]) -> Tuple[Tensor, ...]:
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

    task_schedule: Sequence[Set[int]]
    """A sequence of sets containing the classes for each task.

    In online continual learning your learner may not have access to this
    attribute. It is provided for evaluation and debugging.
    """

    num_classes: int
    """The number of classes in the dataset."""

    default_task_count: int
    """The default number of tasks in the dataset."""

    mean: Optional[Sequence[float]]
    """The mean of the features in the dataset used for normalization."""

    std: Optional[Sequence[float]]
    """The standard deviation of the features in the dataset used for normalization."""

    default_train_transform: Optional[Callable[[Any], Tensor]] = ToTensor()
    """The default transform to apply to the dataset."""

    default_test_transform: Optional[Callable[[Any], Tensor]] = ToTensor()
    """The default transform to apply to the dataset."""

    schema: Schema
    """A schema describing the format of the data."""

    stream: Stream[LabeledInstance]
    """Stream containing each task in sequence."""

    shape: Sequence[int]
    """The shape of each input example."""

    task_mask: Tensor
    """A mask for the output for each task of shape (num_tasks, num_classes)"""

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

        if num_tasks is None:
            num_tasks = self.default_task_count
        if train_transform is None:
            train_transform = self.default_train_transform
        if test_transform is None:
            test_transform = self.default_test_transform
        if normalize_features and self.mean is not None and self.std is not None:
            normalize = Normalize(self.mean, self.std)
            # If transforms are provided, compose them with the normalization
            # transform. Otherwise, just use the normalization transform.
            train_transform = (
                Compose([train_transform, normalize]) if train_transform else normalize
            )
            test_transform = (
                Compose([test_transform, normalize]) if test_transform else normalize
            )
        elif normalize_features:
            raise ValueError(
                "Cannot normalize features since mean and std are not defined."
            )
        self.num_tasks = num_tasks

        generator = torch.Generator().manual_seed(seed)
        self.task_schedule = class_incremental_schedule(
            self.num_classes,
            num_tasks,
            shuffle=shuffle_tasks,
            generator=generator,
        )

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
        self.stream = TorchStream.from_classification(
            ConcatDataset(self.train_tasks),
            num_classes=self.num_classes,
            shuffle=False,
            dataset_name=str(self),
            shape=self.shape,
        )
        self.schema = self.stream.get_schema()
        self.task_mask = class_schedule_to_task_mask(
            self.task_schedule,
            self.num_classes,
        )

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
        return _PreloadedDataset(torch.stack(xs), torch.tensor(ys))

    @classmethod
    @abstractmethod
    def _download_dataset(
        cls,
        train: bool,
        directory: Path,
        auto_download: bool,
        transform: Optional[Any],
        target_transform: Optional[Callable[[Any], Any]] = None,
    ) -> Dataset[Tuple[Tensor, Tensor]]:
        pass

    def __str__(self) -> str:
        return f"{self.__class__.__name__}{self.num_classes}/{self.num_tasks}"

    def train_loaders(
        self,
        batch_size: int,
        shuffle: bool = False,
        **kwargs: Any,
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
        return cast(
            List[DataLoader[Tuple[Tensor, Tensor]]],
            [
                DataLoader(
                    task,
                    batch_size=batch_size,
                    shuffle=shuffle,
                    **kwargs,
                    collate_fn=getattr(task, "collate_fn", None),
                )
                for task in self.train_tasks
            ],
        )

    def test_loaders(
        self,
        batch_size: int,
        **kwargs: Any,
    ) -> Sequence[DataLoader[Tuple[Tensor, Tensor]]]:
        """Get the training streams for the scenario.

        :param batch_size: Collects vectors in batches of this size.
        :param kwargs: Additional keyword arguments to pass to the DataLoader.
        :return: A data loader for each task.
        """
        return cast(
            List[DataLoader[Tuple[Tensor, Tensor]]],
            [
                DataLoader(
                    task,
                    batch_size=batch_size,
                    shuffle=False,
                    **kwargs,
                    collate_fn=getattr(task, "collate_fn", None),
                )
                for task in self.test_tasks
            ],
        )


class _TorchVisionDownload:
    """Shared torchvision dataset downloader for classification scenarios."""

    dataset_type: Type[Dataset]

    @classmethod
    def _download_dataset(
        cls,
        train: bool,
        directory: Path,
        auto_download: bool,
        transform: Optional[Any],
        target_transform: Optional[Callable[[Any], Any]] = None,
    ) -> Dataset[Tuple[Tensor, Tensor]]:
        dataset_type = cast(Any, cls.dataset_type)
        return dataset_type(
            directory,
            train=train,
            download=auto_download,
            transform=transform,
            target_transform=target_transform,
        )


class _BuiltInRotatedDomainScenario(_BuiltInCIScenario):
    """Base class for domain-incremental scenarios with fixed task rotations."""

    default_rotation_max = 180.0

    def __init__(
        self,
        num_tasks: Optional[int] = None,
        rotations: Optional[Sequence[float]] = None,
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
        if num_tasks is None:
            num_tasks = self.default_task_count
        if num_tasks <= 0:
            raise ValueError("Number of tasks should be greater than 0")

        if train_transform is None:
            train_transform = self.default_train_transform
        if test_transform is None:
            test_transform = self.default_test_transform

        if rotations is None:
            step = self.default_rotation_max / num_tasks
            rotations = tuple(i * step for i in range(num_tasks))
        elif len(rotations) != num_tasks:
            raise ValueError("Length of `rotations` must match `num_tasks`")

        normalize = None
        if normalize_features and self.mean is not None and self.std is not None:
            normalize = Normalize(self.mean, self.std)
        elif normalize_features:
            raise ValueError(
                "Cannot normalize features since mean and std are not defined."
            )

        self.num_tasks = num_tasks
        self.rotations = tuple(float(angle) for angle in rotations)
        all_classes = set(range(self.num_classes))
        self.task_schedule = [set(all_classes) for _ in range(self.num_tasks)]

        generator = torch.Generator().manual_seed(seed)
        self.train_tasks = []
        self.test_tasks = []
        for angle in self.rotations:
            train_dataset = self._download_dataset(
                True,
                directory,
                auto_download,
                self._task_transform(angle, train_transform, normalize),
            )
            if shuffle_data:
                train_dataset = self._shuffle_dataset(train_dataset, generator)

            test_dataset = self._download_dataset(
                False,
                directory,
                auto_download,
                self._task_transform(angle, test_transform, normalize),
            )
            self.train_tasks.append(train_dataset)
            self.test_tasks.append(test_dataset)

        if preload_train:
            self.train_tasks = self._preload_datasets(self.train_tasks)
        if preload_test:
            self.test_tasks = self._preload_datasets(self.test_tasks)

        self.stream = TorchStream.from_classification(
            ConcatDataset(self.train_tasks),
            num_classes=self.num_classes,
            shuffle=False,
            dataset_name=str(self),
            shape=self.shape,
        )
        self.schema = self.stream.get_schema()
        self.task_mask = class_schedule_to_task_mask(
            self.task_schedule,
            self.num_classes,
        )

    @staticmethod
    def _task_transform(
        angle: float,
        base_transform: Optional[Callable[[Any], Tensor]],
        normalize: Optional[Normalize],
    ) -> Callable[[Any], Tensor]:
        transforms: list[Callable[[Any], Any]] = []
        if base_transform is not None:
            transforms.append(base_transform)
        transforms.append(Lambda(lambda x, a=angle: rotate(x, a)))
        if normalize is not None:
            transforms.append(normalize)
        return Compose(transforms)

    @staticmethod
    def _shuffle_dataset(
        dataset: Dataset[Tuple[Tensor, Tensor]],
        generator: torch.Generator,
    ) -> Dataset[Tuple[Tensor, Tensor]]:
        targets = get_targets(dataset)
        indices = torch.randperm(len(targets), generator=generator)
        subset = torch.utils.data.Subset(dataset, cast(Sequence[int], indices))
        subset.targets = targets[indices]  # type: ignore
        return subset
