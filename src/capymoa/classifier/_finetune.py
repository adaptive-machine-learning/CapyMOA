from typing import Any

import torch
from torch import Tensor, device, nn, optim
from torch.optim.optimizer import Optimizer

from capymoa.base import BatchClassifier, TorchMixin
from capymoa.stream import Schema


class _FinetuneModule(nn.Module):
    """Torch state of Finetune."""

    def __init__(self, model: nn.Module, criterion: nn.Module) -> None:
        super().__init__()
        self.model = model
        self.criterion = criterion


class Finetune(TorchMixin[_FinetuneModule], BatchClassifier):
    """Finetune a PyTorch neural network using stochastic gradient descent.

    >>> from capymoa.datasets import ElectricityTiny
    >>> from capymoa.classifier import evaluate_classifier
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
    ...     random_seed=0,
    ... )
    >>> results = evaluate_classifier(stream, learner, batch_size=32)
    >>> print(f"{results['cumulative']['accuracy']:.1f}")
    62.4

    Alternatively, you can use a custom model and optimizer:

    >>> model = nn.Sequential(nn.Linear(6, 10), nn.ReLU(), nn.Linear(10, 2))
    >>> optimizer = Adam(model.parameters(), lr=0.001)
    >>> learner = Finetune(
    ...     schema=stream.get_schema(),
    ...     model=model,
    ...     optimizer=optimizer,
    ...     random_seed=0,
    ... )
    >>> results = evaluate_classifier(stream, learner, batch_size=32)
    >>> print(f"{results['cumulative']['accuracy']:.1f}")
    60.4

    """

    def __init__(
        self,
        schema: Schema,
        model: nn.Module | type[nn.Module],
        optimizer: Optimizer | type[Optimizer] = optim.Adam,
        model_params: dict[str, Any] | None = None,
        optimizer_params: dict[str, Any] | None = None,
        criterion: nn.Module | None = None,
        device: device | str = "cpu",
        random_seed: int | None = None,
    ) -> None:
        """Construct a learner to finetune a neural network.

        :param schema: Describes streaming data types and shapes.
        :param model: A classifier model that takes a ``(bs, input_dim)`` matrix
            and returns a ``(bs, num_classes)`` matrix. Alternatively, a model
            class (e.g. :class:`~capymoa.core.torch.ann.Perceptron`),
            constructed with ``schema`` and ``model_params``.
        :param optimizer: A PyTorch gradient descent optimizer. Alternatively,
            an optimizer class (e.g. :class:`torch.optim.Adam`), constructed
            with the model's parameters and ``optimizer_params``.
        :param model_params: Extra keyword arguments passed when constructing
            ``model`` from a class. Ignored if ``model`` is already a
            model instance.
        :param optimizer_params: Extra keyword arguments passed when
            constructing ``optimizer`` from a class. Ignored if
            ``optimizer`` is already an optimizer instance.
        :param criterion: Loss function to use for training. Defaults to
            :class:`torch.nn.CrossEntropyLoss`.
        :param device: Hardware for training. Move it later with
            ``learner.module.to(...)``. Your model is at ``learner.module.model``.
        :param random_seed: Seeds torch :py:func:`torch.manual_seed`.
        """
        super().__init__(schema, random_seed)
        if criterion is None:
            criterion = nn.CrossEntropyLoss()
        # seed for reproducibility
        torch.manual_seed(self.random_seed)
        if not isinstance(model, nn.Module):
            model = model(schema, **(model_params or {}))
        self._module = _FinetuneModule(model, criterion).to(device)
        self.x_dtype = next(model.parameters(), torch.empty(0)).dtype
        self._optimizer: Optimizer = (
            optimizer
            if isinstance(optimizer, Optimizer)
            else optimizer(model.parameters(), **(optimizer_params or {}))
        )

    def batch_train(self, x: Tensor, y: Tensor) -> None:
        self._module.model.train()
        self._optimizer.zero_grad()

        # Forward pass
        y_hat = self._module.model(x)

        # Compute loss
        loss: Tensor = self._module.criterion(y_hat, y)

        # Backward pass and optimization
        loss.backward()
        self._optimizer.step()
        assert not loss.isnan(), "Loss is NaN"

    @torch.no_grad()
    def batch_predict_proba(self, x: Tensor) -> Tensor:
        """Predict the probabilities of the classes for the given batch of data.

        :param x: Input data of shape (batch_size, num_features).
        :param y: Target labels of shape (batch_size,).
        :return: Predicted probabilities of shape (batch_size, num_classes).
        """
        self._module.model.eval()
        return self._module.model(x).softmax(dim=1)

    def __str__(self) -> str:
        model_name = str(self._module.model.__class__.__name__)
        optimizer_name = str(self._optimizer.__class__.__name__)
        return f"Finetune(model={model_name}, optimizer={optimizer_name})"
