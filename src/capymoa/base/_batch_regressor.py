"""Batch regressor base class.

Separated from :mod:`capymoa.base._regressor` so that importing
:mod:`capymoa.base` does not import PyTorch. :class:`BatchRegressor` cannot work
without PyTorch, so it is exposed lazily from :mod:`capymoa.base`.
"""

from abc import ABC, abstractmethod

import numpy as np
import torch
from torch import Tensor

from capymoa.core import RegressionInstance
from capymoa.core import TargetValue

from ._batch import Batch
from ._regressor import Regressor


class BatchRegressor(Regressor, Batch, ABC):
    """Base class for regressor that support mini-batches.

    Supported by:

    - :func:`capymoa.evaluation.prequential_evaluation`

    Evaluators that support batch classifiers will call the :func:`batch_train`
    and :func:`batch_predict` methods instead of :func:`train` and
    :func:`predict`:

    >>> from capymoa.base import BatchRegressor
    >>> from capymoa.datasets import FriedTiny
    >>> from capymoa.evaluation import prequential_evaluation
    >>>
    >>> batch_size = 500
    >>> class MyBatchRegressor(BatchRegressor):
    ...     def batch_train(self, x, y):
    ...         print(f"batch_train x: {x.shape} {x.dtype}")
    ...         print(f"batch_train y: {y.shape} {y.dtype}")
    ...
    ...     def batch_predict(self, x):
    ...         print(f"batch_predict x: {x.shape} {x.dtype}")
    ...         return np.zeros((x.shape[0],))
    ...
    >>> stream = FriedTiny()
    >>> learner = MyBatchRegressor(stream.get_schema())
    >>> _ = prequential_evaluation(
    ...     stream,
    ...     learner,
    ...     batch_size=batch_size,
    ...     max_instances=721
    ... )
    batch_predict x: torch.Size([500, 10]) torch.float32
    batch_train x: torch.Size([500, 10]) torch.float32
    batch_train y: torch.Size([500]) torch.float32
    batch_predict x: torch.Size([221, 10]) torch.float32
    batch_train x: torch.Size([221, 10]) torch.float32
    batch_train y: torch.Size([221]) torch.float32

    You can manually use ``itertools.batched`` (python 3.12) function and
    ``np.stack`` to collect batches of instances as a matrix:

    >>> from itertools import islice
    >>> from capymoa._utils import batched # Not available in python < 3.12
    >>> for i, batch in enumerate(batched(stream, 100)):
    ...     x = np.stack([instance.x for instance in batch])
    ...     y = np.stack([instance.y_value for instance in batch])
    ...     x = torch.from_numpy(x).to(dtype=learner.x_dtype, device=learner.device)
    ...     y = torch.from_numpy(y).to(dtype=learner.y_dtype, device=learner.device)
    ...     learner.batch_train(x, y)
    ...     break
    batch_train x: torch.Size([100, 10]) torch.float32
    batch_train y: torch.Size([100]) torch.float32

    The default implementation of :func:`train` and :func:`predict` calls the
    batch variants with a batch of size 1. This is useful for parts of CapyMOA
    that expect a classifier to be able to train and predict on single
    instances.

    >>> instance = next(stream)
    >>> learner.train(instance)
    batch_train x: torch.Size([1, 10]) torch.float32
    batch_train y: torch.Size([]) torch.float32
    >>> learner.predict(instance)
    batch_predict x: torch.Size([1, 10]) torch.float64
    np.float64(0.0)
    """

    x_dtype: torch.dtype = torch.float32
    y_dtype: torch.dtype = torch.float32

    @abstractmethod
    def batch_train(self, x: Tensor, y: Tensor) -> None:
        """Train the classifier with a batch of instances.

        :param x: Batch of :py:attr:`x_dtype` valued feature vectors
            ``(batch_size, num_features)``
        :param y: Batch of :py:attr:`y_dtype` valued targets ``(batch_size,)``.
        """

    @abstractmethod
    def batch_predict(self, x: Tensor) -> Tensor:
        """Return probability estimates for each label in a batch.

        :param x: Batch of :py:attr:`x_dtype` valued feature vectors
            ``(batch_size, num_features)``
        :return: Predicted batch of :py:attr:`y_dtype` valued targets
            ``(batch_size,)``.
        """

    def train(self, instance: RegressionInstance) -> None:
        """Calls :func:`batch_train` with a batch of size 1."""
        x_ = torch.from_numpy(instance.x.reshape(1, -1))
        x_ = x_.to(dtype=self.x_dtype, device=self.device)
        y_ = torch.scalar_tensor(
            instance.y_value, dtype=self.y_dtype, device=self.device
        )
        return self.batch_train(x_, y_)

    def predict(self, instance: RegressionInstance) -> TargetValue:
        """Calls :func:`batch_predict` with a batch of size 1."""
        x_ = torch.from_numpy(instance.x.reshape(1, -1))
        return np.float64(self.batch_predict(x_).item())
