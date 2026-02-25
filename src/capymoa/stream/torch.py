from typing import Optional, Sequence, Tuple

import torch

from capymoa.stream import Stream, Schema
from capymoa.instance import LabeledInstance, RegressionInstance
from torch.utils.data import Dataset


def _shuffle_dataset(dataset: Dataset, seed: Optional[int] = None) -> Dataset:
    rng = torch.Generator()
    if seed is not None:
        rng.manual_seed(seed)
    indicies = torch.randperm(len(dataset), generator=rng)
    return torch.utils.data.Subset(dataset, indicies)


class TorchStream(Stream):
    """A stream adapter for PyTorch datasets.

    This class converts PyTorch datasets into CapyMOA streams for both classification
    and regression tasks.

    Creating a classification stream from a PyTorch dataset:

    >>> from capymoa.datasets import get_download_dir
    >>> from capymoa.stream import TorchStream
    >>> from torchvision import datasets, transforms
    >>>
    >>> dataset = datasets.FashionMNIST(
    ...     root=get_download_dir(),
    ...     train=True,
    ...     download=True,
    ...     transform=transforms.ToTensor()
    ... )  # doctest: +SKIP
    >>> stream = TorchStream.from_classification(
    ...     dataset, num_classes=10, class_names=dataset.classes
    ... )  # doctest: +SKIP
    >>> stream.next_instance()  # doctest: +SKIP
    LabeledInstance(...)

    Creating a shuffled classification stream:

    >>> import torch
    >>> from torch.utils.data import TensorDataset
    >>>
    >>> dataset = TensorDataset(
    ...     torch.tensor([[1.0], [2.0], [3.0]]),
    ...     torch.tensor([0, 1, 2])
    ... )
    >>> stream = TorchStream.from_classification(
    ...     dataset, num_classes=3, shuffle=True, shuffle_seed=0
    ... )
    >>> [float(inst.x[0]) for inst in stream]
    [3.0, 1.0, 2.0]

    Streams can be restarted to iterate again:

    >>> stream.restart()
    >>> [float(inst.x[0]) for inst in stream]
    [3.0, 1.0, 2.0]

    Creating a regression stream:

    >>> dataset = TensorDataset(
    ...     torch.tensor([[1.0], [2.0], [3.0]]),
    ...     torch.tensor([0.5, 1.5, 2.5])
    ... )
    >>> stream = TorchStream.from_regression(
    ...     dataset, shuffle=True, shuffle_seed=0
    ... )
    >>> [(float(inst.x[0]), float(inst.y_value)) for inst in stream]
    [(3.0, 2.5), (1.0, 0.5), (2.0, 1.5)]
    """

    @staticmethod
    def from_regression(
        dataset: Dataset[Tuple[torch.Tensor, torch.Tensor | float]],
        dataset_name: str = "TorchStream",
        shuffle: bool = False,
        shuffle_seed: Optional[int] = None,
    ) -> "TorchStream":
        """Construct a stream for regression from a PyTorch Dataset.

        :param dataset: A PyTorch Dataset that yields tuples of (features, target) for
            regression tasks.
        :param dataset_name: An optional name for the stream.
        :param shape: An optional shape for the features. If not provided, features will
            be treated as flat vectors.
        :param shuffle: Whether to shuffle the dataset.
        :param shuffle_seed: An optional seed for shuffling the dataset.
        :return: A TorchStream instance for regression.
        """

        # Construct the schema based on the dataset and provided parameters
        X, _ = dataset[0]
        n_features = X.numel()
        features = [str(f) for f in range(n_features)] + ["target"]
        schema = Schema.from_custom(
            features=features,
            target="target",
            name=dataset_name,
        )

        dataset = _shuffle_dataset(dataset, seed=shuffle_seed) if shuffle else dataset
        return TorchStream(dataset, schema)

    @staticmethod
    def from_classification(
        dataset: Dataset[Tuple[torch.Tensor, torch.Tensor | int]],
        num_classes: int,
        class_names: Optional[Sequence[str]] = None,
        dataset_name: str = "TorchStream",
        shape: Optional[Sequence[int]] = None,
        shuffle: bool = False,
        shuffle_seed: Optional[int] = None,
    ) -> "TorchStream":
        """Construct a stream for classification from a PyTorch Dataset.

        :param dataset: A PyTorch Dataset that yields tuples of (features, target).
        :param num_classes: The number of classes in the classification task.
        :param class_names: An optional sequence of class names corresponding to the class indices.
        :param dataset_name: An optional name for the stream.
        :param shape: An optional shape for the features. If not provided, features will
            be treated as flat vectors.
        :param shuffle: Whether to shuffle the dataset.
        :param shuffle_seed: An optional seed for shuffling the dataset.
        :return: A TorchStream instance.
        """

        if class_names is None:
            class_names = [str(k) for k in range(num_classes)]
        if len(class_names) != num_classes:
            raise ValueError("Length of class_names must match num_classes.")

        # Construct the schema based on the dataset and provided parameters
        X, _ = dataset[0]
        n_features = X.numel()
        features = [str(f) for f in range(n_features)] + ["class"]
        schema = Schema.from_custom(
            features=features,
            target="class",
            categories={"class": class_names},
            name=dataset_name,
        )
        schema._shape = shape if shape is not None else (n_features,)

        dataset = _shuffle_dataset(dataset, seed=shuffle_seed) if shuffle else dataset
        return TorchStream(dataset, schema)

    def __init__(
        self,
        dataset: Dataset,
        schema: Schema,
    ):
        """Construct a TorchStream from a PyTorch Dataset and a Schema.

        Usually you want :meth:`from_classification` or :meth:`from_regression`.

        :param dataset: A PyTorch Dataset that yields tuples of (features, target).
        :param schema: A Schema object that describes the structure of the data,
            including feature names and target information.
        """
        self._dataset = dataset
        self.schema = schema
        self._index = 0

    def has_more_instances(self):
        return len(self._dataset) > self._index

    def next_instance(self):
        if not self.has_more_instances():
            raise StopIteration()

        X, y = self._dataset[self._index]
        self._index += 1  # increment counter for next call

        if self.schema.is_classification():
            # Tensors on the CPU and NumPy arrays share their underlying memory locations
            # We should prefer numpy over tensors in instances to improve compatibility
            # See: https://pytorch.org/tutorials/beginner/blitz/tensor_tutorial.html#bridge-to-np-label
            X = X.view(-1).numpy()
            if isinstance(y, torch.Tensor) and torch.isnan(y):
                y = -1
            return LabeledInstance.from_array(self.schema, X, int(y))
        elif self.schema.is_regression():
            X = X.view(-1).numpy()
            y = y.item()  # Convert single-value tensor to a Python scalar
            return RegressionInstance.from_array(self.schema, X, y)
        else:
            raise ValueError("Schema must be either classification or regression.")

    def get_schema(self):
        return self.schema

    def get_moa_stream(self):
        return None

    def restart(self):
        self._index = 0

    def __len__(self) -> int:
        return len(self._dataset)
