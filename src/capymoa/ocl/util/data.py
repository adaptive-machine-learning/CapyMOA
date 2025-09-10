"""Utilities for continual learning when using PyTorch datasets."""

from typing import cast, Sequence, Sized, Set, Tuple

import torch
from torch import BoolTensor, LongTensor, Tensor
from torch.utils.data import Dataset, Subset


def _is_unique_consecutive_from_zero(tensor: Tensor) -> bool:
    """Check if a tensor contains unique consecutive integers starting from 0.

    >>> _is_unique_consecutive_from_zero(torch.tensor([0, 1, 2]))
    True
    >>> _is_unique_consecutive_from_zero(torch.tensor([0, 1, 3]))
    False
    >>> _is_unique_consecutive_from_zero(torch.tensor([0, 1, 1]))
    False

    :param tensor: The tensor to check.
    :return: True if the tensor contains unique consecutive integers starting from 0,
    """
    unique_values: Tensor = tensor.unique()  # type: ignore
    return tensor.numel() == unique_values.numel() and unique_values.equal(
        torch.arange(unique_values.numel())
    )


def get_targets(dataset: Dataset[Tuple[Tensor, Tensor]]) -> LongTensor:
    """Return the targets of a dataset as a 1D tensor.

    * If the dataset has a `targets` attribute, it is used.
    * Otherwise, the targets are extracted from the dataset by iterating over it.

    :param dataset: The dataset to get the targets from.
    :return: A 1D tensor containing the targets of the dataset.
    """
    if not isinstance(dataset, Sized):
        raise ValueError("Dataset should implement the `Sized` protocol")

    # If possible use the dataset's targets
    if hasattr(dataset, "targets") and isinstance(dataset.targets, torch.Tensor):
        assert dataset.targets.dim() == 1, "Targets should be a 1D tensor"
        return LongTensor(dataset.targets)
    if hasattr(dataset, "targets") and isinstance(dataset.targets, list):
        return LongTensor(dataset.targets)

    # Otherwise loop over the dataset to get the labels
    labels = LongTensor(len(dataset))
    for i, (_, y) in enumerate(dataset):  # type: ignore
        labels[i] = int(y)  # type: ignore
    return labels


def get_class_indices(targets: LongTensor) -> dict[int, LongTensor]:
    """Return a dictionary containing the indices of each sample given the class.

    >>> targets = torch.tensor([0, 1, 0, 1, 2])
    >>> get_class_indices(targets)
    {0: tensor([0, 2]), 1: tensor([1, 3]), 2: tensor([4])}

    :param targets: A 1D tensor containing the class labels.
    :return: A dictionary containing the indices of each class.
    """
    indices: dict[int, LongTensor] = {}
    unique_labels = cast(Tensor, targets.unique())  # type: ignore
    for label in unique_labels:
        mask = targets.eq(label)
        indices[int(label)] = torch.nonzero(mask).squeeze(1).long()  # type: ignore
    return indices


def partition_by_schedule(
    dataset: Dataset[Tuple[Tensor, Tensor]],
    class_schedule: Sequence[Set[int]],
    shuffle: bool = False,
    rng: torch.Generator = torch.default_generator,
) -> Sequence[Dataset[Tuple[Tensor, Tensor]]]:
    """Divide a dataset into multiple datasets based on a class schedule.

    In class incremental learning, a task is a dataset containing a subset of
    the classes in the original dataset. This function divides a dataset into
    multiple tasks, each containing a subset of the classes.

    :param dataset: The dataset to divide.
    :param class_schedule: A sequence of sets containing class indices defining
        task order and composition.
    :param shuffle: If True, the samples in each task are shuffled.
    :param rng: The random number generator used for shuffling, defaults
        to torch.default_generator
    :return: A list of datasets, each corresponding to a task.
    """
    targets = get_targets(dataset)
    class_indices = get_class_indices(targets)
    task_datasets = []
    for classes in class_schedule:
        indices = torch.cat([class_indices[c] for c in classes])
        if shuffle:
            indices = indices[torch.randperm(indices.numel(), generator=rng)]
        subset = Subset(dataset, cast(Sequence[int], indices))
        subset.targets = targets[indices]  # type: ignore
        assert isinstance(subset, Sized), "Subset should be a Sized object"
        task_datasets.append(subset)

    return task_datasets


def class_incremental_split(
    dataset: Dataset[Tuple[Tensor, Tensor]],
    num_tasks: int,
    shuffle_tasks: bool = True,
    generator: torch.Generator = torch.default_generator,
) -> tuple[Sequence[Dataset[Tuple[Tensor, Tensor]]], Sequence[Set[int]]]:
    """Divide a dataset into multiple tasks for class incremental learning.

    >>> from torch.utils.data import TensorDataset
    >>> x = torch.tensor([[1, 2], [3, 4], [5, 6], [7, 8]])
    >>> y = torch.tensor([0, 1, 2, 3])
    >>> dataset = TensorDataset(x, y)
    >>> tasks, schedule = class_incremental_split(dataset, 2, shuffle_tasks=False)
    >>> schedule
    [{0, 1}, {2, 3}]
    >>> tasks[0][0]
    (tensor([1, 2]), tensor(0))
    >>> tasks[1][0]
    (tensor([5, 6]), tensor(2))


    :param dataset: The dataset to divide.
    :param num_tasks: The number of tasks to divide the dataset into.
    :param shuffle_tasks: When False, the classes occur in numerical order of their
        labels. When True, the classes are shuffled.
    :param generator: The random number generator used for shuffling, defaults
        to torch.default_generator
    :return: A tuple containing the list of tasks and the class schedule.
    """
    targets = get_targets(dataset)
    unique_labels: Tensor = targets.unique()  # type: ignore
    if not _is_unique_consecutive_from_zero(unique_labels):
        raise ValueError("Labels should be consecutive integers starting from 0")
    num_classes = unique_labels.numel()
    class_schedule = class_incremental_schedule(
        num_classes, num_tasks, shuffle_tasks, generator
    )
    return partition_by_schedule(dataset, class_schedule), class_schedule


def class_incremental_schedule(
    num_classes: int,
    num_tasks: int,
    shuffle: bool = True,
    generator: torch.Generator = torch.default_generator,
) -> Sequence[Set[int]]:
    """Returns a class schedule for class incremental learning.

    >>> class_incremental_schedule(9, 3, shuffle=False)
    [{0, 1, 2}, {3, 4, 5}, {8, 6, 7}]

    >>> class_incremental_schedule(9, 3, generator=torch.Generator().manual_seed(0))
    [{8, 0, 2}, {1, 3, 7}, {4, 5, 6}]

    :param num_classes: The number of classes in the dataset.
    :param num_tasks: The number of tasks to divide the classes into.
    :param shuffle: When False, the classes occur in numerical order of their
        labels. When True, the classes are shuffled.
    :param generator: The random number generator used for shuffling, defaults
        to torch.default_generator
    :return: A list of lists of classes for each task.
    """
    if num_classes < num_tasks:
        raise ValueError("Cannot split classes into more tasks than classes")
    if num_classes == 0 or num_tasks == 0 or num_classes % num_tasks != 0:
        raise ValueError("Number of classes should be divisible by the number of tasks")

    classes = torch.arange(num_classes)
    if shuffle:
        classes = classes[torch.randperm(num_classes, generator=generator)]

    task_size = num_classes // num_tasks
    return [
        set(classes[i : i + task_size].tolist())
        for i in range(0, num_classes, task_size)
    ]


def class_schedule_to_task_mask(
    class_schedule: Sequence[Set[int]], num_classes: int
) -> BoolTensor:
    """Convert a class schedule to a list of boolean masks.

    This is useful when implementing multi-headed neural networks for task
    incremental learning.

    >>> class_schedule_to_task_mask([{0, 1}, {2, 3}], 4)
    tensor([[ True,  True, False, False],
            [False, False,  True,  True]])

    :param num_classes: The total number of classes.
    :param class_schedule: A sequence of sets containing class indices defining
        task order and composition.
    :return: A boolean mask of shape (num_tasks, num_classes)
    """
    min_class = min(map(min, class_schedule), default=-1)
    max_class = max(map(max, class_schedule), default=-1)
    if not 0 <= min_class < num_classes or not 0 <= max_class < num_classes:
        raise ValueError(
            "Classes in the schedule should be within the range of num_classes"
        )

    task_mask = torch.zeros(len(class_schedule), num_classes, dtype=torch.bool)
    for i, classes in enumerate(class_schedule):
        task_mask[i, list(classes)] = True
    return BoolTensor(task_mask)
