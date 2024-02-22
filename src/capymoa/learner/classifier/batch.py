from abc import ABC, abstractmethod, ABCMeta
import numpy as np
from capymoa.learner import ClassifierSSL
from capymoa.stream.stream import Schema, Instance
from numpy.typing import NDArray
from typing import Any


class BatchClassifierSSL(ClassifierSSL, ABC):
    def __init__(self, batch_size: int, schema: Schema = None, random_seed=1):
        super().__init__(schema=schema, random_seed=random_seed)
        self.batch_size = batch_size
        self._features_batch: NDArray[np.float_] = np.zeros(
            (batch_size, schema.get_num_attributes())
        )
        self._class_values_batch: NDArray[Any] = np.zeros(batch_size, dtype=object)
        self._class_indices_batch: NDArray[np.int_] = np.zeros(
            batch_size, dtype=np.int_
        )
        self.batch_idx = 0

        assert -1 not in schema.get_label_values(), "-1 must not be a valid label"

    @abstractmethod
    def train_on_batch(
        self,
        features: NDArray[np.float_],
        class_values: NDArray[Any],
        class_indices: NDArray[np.int_],
    ):
        """Train the model on a batch of instances. Some of the instances
        may be unlabeled, this is coded as a -1 in the y array.

        :param x: Batched instances of shape (batch_size, num_attributes)
        :param y: Batched label vector of shape (batch_size,)
        """
        pass

    def train(self, instance: Instance):
        """Add an instance to the batch and train the model if the batch is full."""
        self._consume(instance.x(), instance.y(), instance.y_index())

    def train_on_unlabeled(self, instance: Instance):
        """Add an unlabeled instance to the batch and train the model if the batch is full."""
        self._consume(instance.x(), None, -1)

    def _consume(self, x: np.ndarray, y, y_index):
        """Add an instance to the batch."""
        self._features_batch[self.batch_idx] = x.astype(np.float_)
        self._class_values_batch[self.batch_idx] = y
        self._class_indices_batch[self.batch_idx] = y_index
        self.batch_idx += 1

        if self.batch_idx == self.batch_size:
            self.train_on_batch(
                self._features_batch,
                self._class_values_batch,
                self._class_indices_batch,
            )
            self.batch_idx = 0
