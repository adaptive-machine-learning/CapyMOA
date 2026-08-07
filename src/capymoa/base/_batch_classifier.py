"""Batch classifier base class.

Separated from :mod:`capymoa.base._classifier` so that importing
:mod:`capymoa.base` does not import PyTorch. :class:`BatchClassifier` cannot
work without PyTorch, so it is exposed lazily from :mod:`capymoa.base`.
"""

from abc import ABC, abstractmethod
from typing import Optional

import numpy as np
import torch

from capymoa.instance import Instance, LabeledInstance
from capymoa.type_alias import LabelProbabilities

from ._batch import Batch
from ._classifier import Classifier


class BatchClassifier(Classifier, Batch, ABC):
    """Base class for classifiers that support mini-batches.

    Supported by:

    - :func:`capymoa.ocl.evaluation.ocl_train_eval_loop`
    - :func:`capymoa.evaluation.prequential_evaluation`

    Evaluators that support batch classifiers will call the :func:`batch_train`
    and :func:`batch_predict_proba` methods instead of :func:`train` and
    :func:`predict_proba`:

    >>> from capymoa.base import BatchClassifier
    >>> from capymoa.datasets import ElectricityTiny
    >>> from capymoa.evaluation import prequential_evaluation
    >>>
    >>> batch_size = 500
    >>> class MyBatchClassifier(BatchClassifier):
    ...     def batch_train(self, x, y):
    ...         print(f"batch_train x: {x.shape} {x.dtype}")
    ...         print(f"batch_train y: {y.shape} {y.dtype}")
    ...
    ...     def batch_predict_proba(self, x):
    ...         print(f"batch_predict_proba x: {x.shape} {x.dtype}")
    ...         return torch.zeros((x.shape[0], self.schema.get_num_classes()))
    ...
    >>> stream = ElectricityTiny()
    >>> learner = MyBatchClassifier(stream.get_schema())
    >>> _ = prequential_evaluation(
    ...     stream,
    ...     learner,
    ...     batch_size=batch_size,
    ...     max_instances=721
    ... )
    batch_predict_proba x: torch.Size([500, 6]) torch.float32
    batch_train x: torch.Size([500, 6]) torch.float32
    batch_train y: torch.Size([500]) torch.int64
    batch_predict_proba x: torch.Size([221, 6]) torch.float32
    batch_train x: torch.Size([221, 6]) torch.float32
    batch_train y: torch.Size([221]) torch.int64

    You can manually use ``itertools.batched`` (python 3.12) function and
    ``np.stack`` to collect batches of instances as a matrix:

    >>> from itertools import islice
    >>> from capymoa._utils import batched # Not available in python < 3.12
    >>> stream.restart() # streams are stateful, so restart it
    >>> for i, batch in enumerate(batched(stream, 100)):
    ...     x = np.stack([instance.x for instance in batch])
    ...     y = np.stack([instance.y_index for instance in batch])
    ...     x = torch.from_numpy(x).to(learner.device, learner.x_dtype)
    ...     y = torch.from_numpy(y).to(learner.device, learner.y_dtype)
    ...     learner.batch_train(x, y)
    ...     break
    batch_train x: torch.Size([100, 6]) torch.float32
    batch_train y: torch.Size([100]) torch.int64

    The default implementation of :func:`train` and :func:`predict` calls the
    batch variants with a batch of size 1. This is useful for parts of CapyMOA
    that expect a classifier to be able to train and predict on single
    instances.

    >>> instance = next(stream)
    >>> learner.train(instance)
    batch_train x: torch.Size([1, 6]) torch.float32
    batch_train y: torch.Size([1]) torch.int64
    >>> learner.predict(instance)
    batch_predict_proba x: torch.Size([1, 6]) torch.float32
    0
    >>> learner.predict_proba(instance)
    batch_predict_proba x: torch.Size([1, 6]) torch.float32
    array([0., 0.])
    """

    x_dtype: torch.dtype = torch.float32
    y_dtype: torch.dtype = torch.int64

    @abstractmethod
    def batch_train(self, x: torch.Tensor, y: torch.Tensor) -> None:
        """Train with a batch of instances.

        :param x: Batch of :py:attr:`x_dtype` valued feature vectors
            ``(batch_size, num_features)``
        :param y: Batch of :py:attr:`y_dtype` valued labels ``(batch_size,)``.
        """

    @abstractmethod
    def batch_predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Predict the probabilities of the classes for a batch of instances.

        :param x: Batch of :py:attr:`x_dtype` valued feature vectors
            ``(batch_size, num_features)``
        :return: Batch of :py:attr:`x_dtype` valued predicted probabilities
            ``(batch_size, num_classes)``.
        """

    def batch_predict(self, x: torch.Tensor) -> torch.Tensor:
        """Predict the labels for a batch of instances.

        :param x: Batch of :py:attr:`x_dtype` valued feature vectors
            ``(batch_size, num_features)``
        :return: Predicted batch of :py:attr:`y_dtype` valued labels
            ``(batch_size,)``.
        """
        return self.batch_predict_proba(x).argmax(1).to(self.device, self.y_dtype)

    def train(self, instance: LabeledInstance) -> None:
        """Calls :func:`batch_train` with a batch of size 1."""
        x = torch.from_numpy(instance.x).view(1, -1)
        x = x.to(self.device, self.x_dtype)
        y = torch.scalar_tensor(
            instance.y_index, dtype=self.y_dtype, device=self.device
        ).view(1)
        return self.batch_train(x, y)

    def predict_proba(self, instance: Instance) -> Optional[LabelProbabilities]:
        """Calls :func:`batch_predict_proba` with a batch of size 1."""
        x = torch.from_numpy(instance.x.reshape(1, -1))
        x = x.to(self.device, self.x_dtype)
        return self.batch_predict_proba(x).flatten().numpy().astype(np.float64)
