from pathlib import Path
from typing import Any, Callable, List, Optional, Sequence, Tuple, cast

import torch
from torch import Tensor
from torch.utils.data import ConcatDataset, Dataset, Subset

from capymoa.datasets import get_download_dir
from capymoa.datasets._torch_utils import TensorDatasetWithTransform
from capymoa.datasets._utils import download_numpy_dataset
from capymoa.ocl.util.data import group_indicies
from capymoa.stream import TorchStream

from ._base import _BuiltInCIScenario
from ._constants import _SOURCES
from ._vision import DomainCIFAR100


class SplitCIFAR100ViT(_BuiltInCIScenario):
    """CIFAR100 encoded by a Vision Transformer (ViT).

    * Encoded using the ``vit_base_patch16_224_augreg_in21k`` pre-trained
      backbone [1]_.
    * 768 dimensional features (extracted from the last layer of the ViT).
    * 100 classes.
    * 50,000 training samples
    * 10,000 testing samples
    * Useful for developing and evaluating prototype based continual
      learning algorithms.

    ..  [1] Model card for ``vit_base_patch16_224.augreg_in21k``
        https://huggingface.co/timm/vit_base_patch16_224.augreg_in21k
    """

    num_classes = 100
    default_task_count = 10
    default_train_transform = None
    default_test_transform = None
    shape = [768]
    _dataset_key = "CIFAR100_vit_base_patch16_224_augreg_in21k"

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
                torch.from_numpy(train_x).float(),
                torch.from_numpy(train_y).long(),
                transform=transform,
                target_transform=target_transform,
            )
        return TensorDatasetWithTransform(
            torch.from_numpy(test_x).float(),
            torch.from_numpy(test_y).long(),
            transform=transform,
            target_transform=target_transform,
        )


class SplitCIFAR10ViT(SplitCIFAR100ViT):
    """CIFAR10 encoded by a Vision Transformer (ViT).

    * Encoded using the ``vit_base_patch16_224_augreg_in21k`` pre-trained
      backbone [1]_.
    * 768 dimensional features (extracted from the last layer of the ViT).
    * 10 classes.
    * 50,000 training samples
    * 10,000 testing samples
    * Useful for developing and evaluating prototype based continual learning
      algorithms.

    ..  [1] Model card for ``vit_base_patch16_224.augreg_in21k``
        https://huggingface.co/timm/vit_base_patch16_224.augreg_in21k
    """

    _dataset_key = "CIFAR10_vit_base_patch16_224_augreg_in21k"

    num_classes = 10
    default_task_count = 5
    shape = [768]


class DomainCIFAR100ViT(SplitCIFAR100ViT):
    """Domain incremental CIFAR-100 ViT variant with 20 classes per task.

    This dataset has exactly 5 tasks. Each task contains one fine-grained class from
    each CIFAR-100 superclass (20 classes per task), while labels are remapped to the 20
    superclass IDs.

    Note that the groupings are subjective based on the original CIFAR-100's coarse
    labels.

    **References:**

    #. Krizhevsky, A. (2009). Learning Multiple Layers of Features from Tiny Images.
    """

    _CIFAR100_CLASS_TO_SUPERCLASS: List[int] = (
        DomainCIFAR100._CIFAR100_CLASS_TO_SUPERCLASS
    )
    _CIFAR100_SUPERCLASS_CLASSES: List[List[int]] = (
        DomainCIFAR100._CIFAR100_SUPERCLASS_CLASSES
    )
    classes = DomainCIFAR100.classes

    num_classes = 20
    default_task_count = 5
    shape = [768]

    def __init__(
        self,
        shuffle_data: bool = True,
        seed: int = 0,
        directory: Path = get_download_dir(),
        auto_download: bool = True,
        train_transform: Optional[Callable[[Any], Tensor]] = None,
        test_transform: Optional[Callable[[Any], Tensor]] = None,
    ):
        """Create the DomainCIFAR100ViT scenario."""
        if train_transform is None:
            train_transform = self.default_train_transform
        if test_transform is None:
            test_transform = self.default_test_transform

        self.num_tasks = self.default_task_count
        all_classes = set(range(self.num_classes))
        self.task_schedule = [set(all_classes) for _ in range(self.num_tasks)]

        generator = torch.Generator().manual_seed(seed)

        def target_transform(y: Any) -> int:
            return self._CIFAR100_CLASS_TO_SUPERCLASS[int(y)]

        train_dataset = self._download_dataset(
            True,
            directory,
            auto_download,
            train_transform,
            target_transform=target_transform,
        )
        test_dataset = self._download_dataset(
            False,
            directory,
            auto_download,
            test_transform,
            target_transform=target_transform,
        )

        superclass_classes = torch.asarray(self._CIFAR100_SUPERCLASS_CLASSES)
        # Shuffle each superclass class order so tasks change with seed.
        if shuffle_data:
            for i, classes in enumerate(superclass_classes):
                superclass_classes[i] = classes[
                    torch.randperm(classes.size(0), generator=generator)
                ]

        self.train_tasks = self._build_domain_tasks(
            train_dataset,
            superclass_classes.T.tolist(),
            shuffle_data,
            generator,
        )
        self.test_tasks = self._build_domain_tasks(
            test_dataset,
            superclass_classes.T.tolist(),
            False,
            generator,
        )

        self.stream = TorchStream.from_classification(
            ConcatDataset(self.train_tasks),
            num_classes=self.num_classes,
            shuffle=False,
            dataset_name=str(self),
            shape=self.shape,
            class_names=self.classes,
        )
        self.schema = self.stream.get_schema()
        self.task_mask = torch.ones(
            (self.num_tasks, self.num_classes), dtype=torch.bool
        )

    @staticmethod
    def _build_domain_tasks(
        dataset: Dataset[Tuple[Tensor, Tensor]],
        task_fine_classes: Sequence[Sequence[int]],
        shuffle_data: bool,
        generator: torch.Generator,
    ) -> List[Dataset[Tuple[Tensor, Tensor]]]:
        targets = cast(torch.LongTensor, torch.asarray(dataset.targets))  # type: ignore
        grouped_indices = group_indicies(
            targets, task_fine_classes, shuffle=shuffle_data, rng=generator
        )
        tasks: List[Dataset[Tuple[Tensor, Tensor]]] = []
        for indices in grouped_indices:
            tasks.append(
                cast(
                    Dataset[Tuple[Tensor, Tensor]],
                    Subset(dataset, indices.tolist()),
                )
            )
        return tasks
