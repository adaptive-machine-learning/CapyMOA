from typing import Callable, Union

import numpy as np
import torch
from torch import Tensor, device, nn, optim
from torch.optim.optimizer import Optimizer, ParamsT

from capymoa.base import BatchClassifier
from capymoa.instance import Instance
from capymoa.stream import Schema
from capymoa.type_alias import LabelProbabilities


class Finetune(BatchClassifier):
    """Finetune a PyTorch neural network using stochastic gradient descent.

    >>> from capymoa.datasets import ElectricityTiny
    >>> from capymoa.evaluation import prequential_evaluation
    >>> from capymoa.classifier import Finetune
    >>> from capymoa.ann import Perceptron
    >>> from torch import nn
    >>> from torch.optim import Adam
    >>> from functools import partial
    >>>
    >>> stream = ElectricityTiny()
    >>> learner = Finetune(
    ...     stream.get_schema(),
    ...     batch_size=16,
    ...     model=Perceptron,
    ...     optimizer=partial(Adam, lr=0.01)
    ... )
    >>> results = prequential_evaluation(stream, learner)
    >>> print(f"{results['cumulative'].accuracy():.1f}")
    65.3

    Alternatively, you can use a custom model and optimizer:

    >>> model = nn.Sequential(nn.Linear(6, 10), nn.ReLU(), nn.Linear(10, 2))
    >>> optimizer = Adam(model.parameters(), lr=0.001)
    >>> learner = Finetune(
    ...     schema=stream.get_schema(),
    ...     model=model,
    ...     optimizer=optimizer,
    ... )
    >>> results = prequential_evaluation(stream, learner)
    >>> print(f"{results['cumulative'].accuracy():.1f}")
    60.4

    """

    def __init__(
        self,
        schema: Schema,
        model: Union[nn.Module, Callable[[Schema], nn.Module]],
        batch_size: int = 32,
        optimizer: Union[Optimizer, Callable[[ParamsT], Optimizer]] = optim.Adam,
        device: Union[device, str] = "cpu",
        criterion: nn.Module = nn.CrossEntropyLoss(),
        random_seed: int = 0,
    ) -> None:
        """Construct a learner to finetune a neural network.

        :param schema: Describes streaming data types and shapes.
        :param model: A classifier model that takes a ``(bs, input_dim)`` matrix
            and returns a ``(bs, num_classes)`` matrix. Alternatively, a
            constructor function that takes a schema and returns a model.
        :param batch_size: Number of samples to use for training in each batch.
        :param optimizer: A PyTorch gradient descent optimizer or a constructor
            function that takes the model parameters and returns an optimizer.
        :param device: Hardware for training.
        :param random_seed: Seeds torch :py:func:`torch.manual_seed`.
        :param criterion: Loss function to use for training.
        """
        super().__init__(schema, batch_size, random_seed)
        # seed for reproducibility
        torch.manual_seed(random_seed)
        #: The model to be trained.
        self.model: nn.Module = model if isinstance(model, nn.Module) else model(schema)
        #: The optimizer to be used for training.
        self.optimizer: Optimizer = (
            optimizer
            if isinstance(optimizer, Optimizer)
            else optimizer(self.model.parameters())
        )
        #: The loss function to be used for training.
        self.criterion: nn.Module = criterion
        #: The device to be used for training.
        self.device: torch.device = torch.device(device)
        #: The data type to convert the input data to.
        self.dtype: torch.dtype = next(self.model.parameters()).dtype

    def batch_train(self, x: np.ndarray, y: np.ndarray) -> None:
        return self.torch_batch_train(
            torch.from_numpy(x).to(self.device, self.dtype),
            torch.from_numpy(y).long().to(self.device),
        )

    def torch_batch_train(self, x: Tensor, y: Tensor) -> None:
        """Train the model on the given batch of data.

        :param x: Input data of shape (batch_size, num_features).
        :param y: Target labels of shape (batch_size,).
        """
        self.model.train()
        self.optimizer.zero_grad()

        # Forward pass
        y_hat = self.model(x)

        # Compute loss
        loss: Tensor = self.criterion(y_hat, y)

        # Backward pass and optimization
        loss.backward()
        self.optimizer.step()
        assert not loss.isnan(), "Loss is NaN"

    @torch.no_grad
    def predict_proba(self, instance: Instance) -> LabelProbabilities:
        self.model.eval()
        x = torch.from_numpy(instance.x).to(self.device, self.dtype)
        x = x.view(1, -1)  # Add batch dimension
        y_hat = self.model(x)
        return y_hat.softmax(dim=1).cpu().numpy()

    @property
    def batch_size(self) -> int:
        return self._batch.batch_size

    def __str__(self) -> str:
        return f"Finetune(model={str(self.model)}, optimizer={str(self.optimizer)}, batch_size={self.batch_size}"
