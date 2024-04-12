from abc import ABC, abstractmethod

import numpy as np
from numpy.typing import NDArray

from capymoa.learner import Classifier
from capymoa.stream.instance import LabeledInstance
from capymoa.stream.stream import Schema
from capymoa.type_alias import FeatureVector


class MiniBatchClassifier(Classifier, ABC):
    """Abstract class for batch classifiers.

    Batch classifiers accumulate instances before processing them in a mini-batch.

    >>> from capymoa.datasets import ElectricityTiny
    >>> class MyBatchClassifier(MiniBatchClassifier):
    ...
    ...     def __str__(self):
    ...         return "MyBatchClassifier"
    ...
    ...     def predict(self, x: FeatureVector):
    ...         return None
    ...
    ...     def predict_proba(self, x: FeatureVector):
    ...         return None
    ...
    ...     def train_on_batch(self, x_batch, y_indices):
    ...         print(f"x_batch.shape = {x_batch.shape}")
    ...         print(f"y_indices.shape = {y_indices.shape}")
    ...         print(x_batch)
    ...         print(y_indices)
    ...
    >>> stream = ElectricityTiny()
    >>> learner = MyBatchClassifier(batch_size=5, schema=stream.schema)
    >>> i = 0
    >>> while stream.has_more_instances():
    ...     instance = stream.next_instance()
    ...     learner.train(instance)
    ...     i += 1
    ...     if i == 10:
    ...         break
    x_batch.shape = (5, 6)
    y_indices.shape = (5,)
    [[0.       0.056443 0.439155 0.003467 0.422915 0.414912]
     [0.021277 0.051699 0.415055 0.003467 0.422915 0.414912]
     [0.042553 0.051489 0.385004 0.003467 0.422915 0.414912]
     [0.06383  0.045485 0.314639 0.003467 0.422915 0.414912]
     [0.085106 0.042482 0.251116 0.003467 0.422915 0.414912]]
    [1 1 1 1 0]
    x_batch.shape = (5, 6)
    y_indices.shape = (5,)
    [[0.106383 0.041161 0.207528 0.003467 0.422915 0.414912]
     [0.12766  0.041161 0.171824 0.003467 0.422915 0.414912]
     [0.148936 0.041161 0.152782 0.003467 0.422915 0.414912]
     [0.170213 0.041161 0.13493  0.003467 0.422915 0.414912]
     [0.191489 0.041161 0.140583 0.003467 0.422915 0.414912]]
    [0 0 0 0 0]
    """
    def __init__(self, batch_size: int, schema: Schema, random_seed=1):
        """Constructor for the BatchClassifier.

        :param batch_size: The size of the batch.
        :param schema: The schema of the stream.
        :param random_seed: Random seed for reproducibility.
        """
        super().__init__(schema=schema, random_seed=random_seed)
        self._batch_size = batch_size
        self._x_batch: NDArray[np.double] = np.zeros((batch_size, schema.get_num_attributes()))
        self._y_batch: NDArray[np.int_] = np.zeros(batch_size, dtype=np.int_)
        self._idx = 0

    @abstractmethod
    def train_on_batch(
        self,
        x_batch: NDArray[np.double],
        y_indices: NDArray[np.int_],
    ):
        """Train the model on a batch of instances.

        :param x_batch: Batched instances of shape (batch_size, num_attributes)
        :param y_indices: Batched label vector of shape (batch_size,)
        """

    def train(self, instance: LabeledInstance):
        """Train the model on a single instance.

        :param instance: The instance to train on.
        """
        self._consume(instance.x, instance.y_index)

    def _consume(self, x: FeatureVector, y_index: LabeledInstance):
        """Add an instance to the batch."""
        self._x_batch[self._idx] = x.astype(float)
        self._y_batch[self._idx] = y_index
        self._idx += 1

        if self._idx == self._batch_size:
            self.train_on_batch(self._x_batch, self._y_batch)
            self._idx = 0
