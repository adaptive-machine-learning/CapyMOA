import copy
from typing import Optional, Sequence, Tuple

import torch

from capymoa.stream import Stream, Schema
from capymoa.stream._stream import _init_moa_stream_and_create_moa_header
from capymoa.instance import (
    LabeledInstance,
)
from torch.utils.data import Dataset


class TorchClassifyStream(Stream[LabeledInstance]):
    """TorchClassifyStream turns a PyTorch dataset into a classification stream.

    >>> from capymoa.evaluation import ClassificationEvaluator
    ...
    >>> from capymoa.datasets import get_download_dir
    >>> from capymoa.stream import TorchClassifyStream
    >>> from torchvision import datasets
    >>> from torchvision.transforms import ToTensor
    >>> print("Using PyTorch Dataset"); pytorchDataset = datasets.FashionMNIST( #doctest:+ELLIPSIS
    ...     root=get_download_dir(),
    ...     train=True,
    ...     download=True,
    ...     transform=ToTensor()
    ... )
    Using PyTorch Dataset...
    >>> pytorch_stream = TorchClassifyStream(pytorchDataset, 10, class_names=pytorchDataset.classes)
    >>> pytorch_stream.get_schema()
    @relation PytorchDataset
    <BLANKLINE>
    @attribute attrib_0 numeric
    @attribute attrib_1 numeric
    ...
    @attribute attrib_783 numeric
    @attribute class {T-shirt/top,Trouser,Pullover,Dress,Coat,Sandal,Shirt,Sneaker,Bag,'Ankle boot'}
    <BLANKLINE>
    @data
    >>> pytorch_stream.next_instance()
    LabeledInstance(
        Schema(PytorchDataset),
        x=[0. 0. 0. ... 0. 0. 0.],
        y_index=9,
        y_label='Ankle boot'
    )

    You can construct :class:`TorchClassifyStream` using a random sampler by passing a sampler
    to the constructor:

    >>> import torch
    >>> from torch.utils.data import RandomSampler, TensorDataset
    >>> dataset = TensorDataset(
    ...     torch.tensor([[1], [2], [3]]), torch.tensor([0, 1, 2])
    ... )
    >>> pytorch_stream = TorchClassifyStream(dataset=dataset, num_classes=3, shuffle=True)
    >>> for instance in pytorch_stream:
    ...     print(instance.x)
    [3]
    [1]
    [2]

    Importantly you can restart the stream to iterate over the dataset in
    the same order again:

    >>> pytorch_stream.restart()
    >>> for instance in pytorch_stream:
    ...     print(instance.x)
    [3]
    [1]
    [2]
    """

    def __init__(
        self,
        dataset: Dataset[Tuple[torch.Tensor, torch.LongTensor]],
        num_classes: int,
        shuffle: bool = False,
        shuffle_seed: int = 0,
        class_names: Optional[Sequence[str]] = None,
        dataset_name: str = "PytorchDataset",
    ):
        """Create a stream from a PyTorch dataset.

        :param dataset: A PyTorch dataset
        :param num_classes: The number of classes in the dataset
        :param shuffle: Randomly sample with replacement, defaults to False
        :param shuffle_seed: Seed for shuffling, defaults to 0
        :param class_names: The names of the classes, defaults to None
        :param dataset_name: The name of the dataset, defaults to "PytorchDataset"
        """
        if not (class_names is None or len(class_names) == num_classes):
            raise ValueError("Number of class labels must match the number of classes")

        self.__init_args_kwargs__ = copy.copy(
            locals()
        )  # save init args for recreation. not a deep copy to avoid unnecessary use of memory

        self._dataset = dataset
        self._index = 0
        self._permutation = torch.arange(len(dataset))

        if shuffle:
            self._permutation = torch.randperm(
                len(dataset),
                generator=torch.Generator().manual_seed(shuffle_seed),
            )

        # Use the first instance to infer the number of attributes
        X, _ = self._dataset[0]
        X_numpy = torch.flatten(X).view(1, -1).detach().numpy()

        # Create a header describing the dataset for MOA
        _, header = _init_moa_stream_and_create_moa_header(
            number_of_instances=X_numpy.shape[0],
            feature_names=[f"attrib_{i}" for i in range(X_numpy.shape[1])],
            values_for_nominal_features={},
            values_for_class_label=class_names or [str(i) for i in range(num_classes)],
            dataset_name=dataset_name,
            target_attribute_name="class",
            target_type="categorical",
        )
        self._schema = Schema(moa_header=header)

    def has_more_instances(self):
        return len(self._dataset) > self._index

    def next_instance(self):
        if not self.has_more_instances():
            return None

        X, y = self._dataset[self._permutation[self._index]]
        self._index += 1  # increment counter for next call

        # Tensors on the CPU and NumPy arrays share their underlying memory locations
        # We should prefer numpy over tensors in instances to improve compatibility
        # See: https://pytorch.org/tutorials/beginner/blitz/tensor_tutorial.html#bridge-to-np-label
        X = X.view(-1).numpy()
        return LabeledInstance.from_array(self._schema, X, y)

    def get_schema(self):
        return self._schema

    def get_moa_stream(self):
        return None

    def restart(self):
        self._index = 0

    def __len__(self) -> int:
        return len(self._dataset)
