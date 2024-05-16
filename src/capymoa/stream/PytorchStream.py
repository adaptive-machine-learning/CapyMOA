import torch

from capymoa.stream import Stream, Schema
from capymoa.stream._stream import _init_moa_stream_and_create_moa_header
from capymoa.instance import (
    LabeledInstance,
    RegressionInstance,
)
from torch.utils.data import Dataset


class PytorchStream(Stream):
    """PytorchStream turns a PyTorch dataset into a datastream.

    >>> from capymoa.evaluation import ClassificationEvaluator
    ...
    >>> from capymoa.datasets import get_download_dir
    >>> from capymoa.stream import PytorchStream
    >>> from torchvision import datasets
    >>> from torchvision.transforms import ToTensor
    >>> print("Using PyTorch Dataset"); pytorchDataset = datasets.FashionMNIST( #doctest:+ELLIPSIS
    ...     root=get_download_dir(),
    ...     train=True,
    ...     download=True,
    ...     transform=ToTensor()
    ... )
    Using PyTorch Dataset...
    >>> pytorch_stream = PytorchStream(dataset=pytorchDataset)
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
        x=ndarray(..., 784),
        y_index=9,
        y_label='Ankle boot'
    )

    """

    # https://pytorch.org/tutorials/beginner/basics/data_tutorial.html
    def __init__(self, dataset: Dataset, enforce_regression=False):
        """Construct PytorchStream from a PyTorch dataset.

        :param dataset: PyTorch containing tuples of `x` and `y`
        :param enforce_regression: Force the task to be a regression task, default is False
        """
        self.training_data = dataset
        # self.train_dataloader = DataLoader(self.training_data, batch_size=1, shuffle=False)
        self.current_instance_index = 0

        X, _ = self.training_data[0]
        X_numpy = torch.flatten(X).view(1, -1).detach().numpy()

        # enforce_regression = np.issubdtype(type(y[0]), np.double)

        self.__moa_stream_with_only_header, self.moa_header = (
            _init_moa_stream_and_create_moa_header(
                number_of_instances=X_numpy.shape[0],
                feature_names=[f"attrib_{i}" for i in range(X_numpy.shape[1])],
                values_for_nominal_features={},
                values_for_class_label=self.training_data.classes,
                dataset_name="PytorchDataset",
                target_attribute_name=None,
                enforce_regression=enforce_regression,
            )
        )

        self.schema = Schema(moa_header=self.moa_header)
        super().__init__(schema=self.schema, CLI=None, moa_stream=None)

    def has_more_instances(self):
        return len(self.training_data) > self.current_instance_index

    def next_instance(self):
        if not self.has_more_instances():
            return None

        X, y = self.training_data[self.current_instance_index]
        self.current_instance_index += 1  # increment counter for next call

        # Tensors on the CPU and NumPy arrays share their underlying memory locations
        # We should prefer numpy over tensors in instances to improve compatibility
        # See: https://pytorch.org/tutorials/beginner/blitz/tensor_tutorial.html#bridge-to-np-label
        X = X.view(-1).numpy()

        if self.schema.is_classification():
            return LabeledInstance.from_array(self.schema, X, y)
        elif self.schema.is_regression():
            return RegressionInstance.from_array(self.schema, X, y)
        else:
            raise ValueError(
                "Unknown machine learning task must be a regression or "
                "classification task"
            )

    def get_schema(self):
        return self.schema

    def get_moa_stream(self):
        raise ValueError("Not a moa_stream, a numpy read file")

    def restart(self):
        self.current_instance_index = 0
