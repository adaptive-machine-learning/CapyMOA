from typing import Any

import torch
from torch import Tensor, device, nn, optim
from torch.optim.optimizer import Optimizer

from capymoa.base import BatchClassifier
from capymoa.core.torch import resolve_model, resolve_optimizer
from capymoa.stream import Schema


class Finetune(BatchClassifier):
    """Finetune a PyTorch neural network using stochastic gradient descent.

    >>> from capymoa.datasets import ElectricityTiny
    >>> from capymoa.evaluation import prequential_evaluation
    >>> from capymoa.classifier import Finetune
    >>> from capymoa.core.torch.ann import Perceptron
    >>> from torch import nn
    >>> from torch.optim import Adam
    >>>
    >>> stream = ElectricityTiny()
    >>> learner = Finetune(
    ...     stream.get_schema(),
    ...     model=Perceptron,
    ...     optimizer=Adam,
    ...     optimizer_params={"lr": 0.01},
    ... )
    >>> results = prequential_evaluation(stream, learner, batch_size=32)
    >>> print(f"{results['cumulative'].accuracy():.1f}")
    62.4

    Alternatively, you can use a custom model and optimizer:

    >>> model = nn.Sequential(nn.Linear(6, 10), nn.ReLU(), nn.Linear(10, 2))
    >>> optimizer = Adam(model.parameters(), lr=0.001)
    >>> learner = Finetune(
    ...     schema=stream.get_schema(),
    ...     model=model,
    ...     optimizer=optimizer,
    ... )
    >>> results = prequential_evaluation(stream, learner, batch_size=32)
    >>> print(f"{results['cumulative'].accuracy():.1f}")
    60.4

    """

    def __init__(
        self,
        schema: Schema,
        model: nn.Module | type[nn.Module] | str,
        optimizer: Optimizer | type[Optimizer] | str = optim.Adam,
        model_params: dict[str, Any] | None = None,
        optimizer_params: dict[str, Any] | None = None,
        criterion: nn.Module | None = None,
        device: device | str = "cpu",
        random_seed: int = 0,
    ) -> None:
        """Construct a learner to finetune a neural network.

        :param schema: Describes streaming data types and shapes.
        :param model: A classifier model that takes a ``(bs, input_dim)`` matrix
            and returns a ``(bs, num_classes)`` matrix. Alternatively, a model
            class (e.g. :class:`~capymoa.core.torch.ann.Perceptron`) or the
            name of one as a string, constructed with ``schema`` and
            ``model_params``.
        :param optimizer: A PyTorch gradient descent optimizer. Alternatively,
            an optimizer class (e.g. :class:`torch.optim.Adam`) or the name of
            one as a string, constructed with the model's parameters and
            ``optimizer_params``.
        :param model_params: Extra keyword arguments passed when constructing
            ``model`` from a class or name. Ignored if ``model`` is already a
            model instance.
        :param optimizer_params: Extra keyword arguments passed when
            constructing ``optimizer`` from a class or name. Ignored if
            ``optimizer`` is already an optimizer instance.
        :param criterion: Loss function to use for training. Defaults to
            :class:`torch.nn.CrossEntropyLoss`.
        :param device: Hardware for training.
        :param random_seed: Seeds torch :py:func:`torch.manual_seed`.
        """
        super().__init__(schema, random_seed)
        if criterion is None:
            criterion = nn.CrossEntropyLoss()
        # seed for reproducibility
        torch.manual_seed(random_seed)
        #: The model to be trained.
        model = resolve_model(model)
        self.model: nn.Module = (
            model
            if isinstance(model, nn.Module)
            else model(schema, **(model_params or {}))
        )
        self.model.to(device)
        #: The optimizer to be used for training.
        optimizer = resolve_optimizer(optimizer)
        self.optimizer: Optimizer = (
            optimizer
            if isinstance(optimizer, Optimizer)
            else optimizer(self.model.parameters(), **(optimizer_params or {}))
        )
        #: The loss function to be used for training.
        self.criterion: nn.Module = criterion
        #: The device to be used for training.
        self.device: torch.device = torch.device(device)
        #: The data type to convert the input data to.
        self.dtype: torch.dtype = next(self.model.parameters()).dtype

    def batch_train(self, x: Tensor, y: Tensor) -> None:
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

    @torch.no_grad()
    def batch_predict_proba(self, x: Tensor) -> Tensor:
        """Predict the probabilities of the classes for the given batch of data.

        :param x: Input data of shape (batch_size, num_features).
        :param y: Target labels of shape (batch_size,).
        :return: Predicted probabilities of shape (batch_size, num_classes).
        """
        self.model.eval()
        return self.model(x).softmax(dim=1)

    def __str__(self) -> str:
        model_name = str(self.model.__class__.__name__)
        optimizer_name = str(self.optimizer.__class__.__name__)
        return f"Finetune(model={model_name}, optimizer={optimizer_name})"
