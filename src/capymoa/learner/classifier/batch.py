from abc import ABC, abstractmethod, ABCMeta
import numpy as np
from capymoa.learner import ClassifierSSL
from capymoa.stream.stream import Schema, Instance


class BatchClassifierSSL(ClassifierSSL, ABC):
    def __init__(self, batch_size: int, schema: Schema = None, random_seed=1):
        super().__init__(schema=schema, random_seed=random_seed)
        self.batch_size = batch_size
        self.batch_x = np.zeros((batch_size, schema.get_num_attributes()))
        self.batch_y = np.zeros(batch_size)
        self.batch_idx = 0

        assert -1 not in schema.get_label_values(), "-1 must not be a valid label"

    @abstractmethod
    def train_on_batch(self, x: np.ndarray, y: np.ndarray):
        """Train the model on a batch of instances. Some of the instances
        may be unlabeled, this is coded as a -1 in the y array.

        :param x: Batched instances of shape (batch_size, num_attributes)
        :param y: Batched label vector of shape (batch_size,)
        """
        pass

    def train(self, instance: Instance):
        """Add an instance to the batch and train the model if the batch is full."""
        self.batch_x[self.batch_idx] = instance.x()
        self.batch_y[self.batch_idx] = instance.y()
        self.batch_idx += 1

        if self.batch_idx == self.batch_size:
            self.train_on_batch(self.batch_x, self.batch_y)
            self.batch_idx = 0

    def train_on_unlabeled(self, instance: Instance):
        """Add an unlabeled instance to the batch and train the model if the batch is full."""
        self.batch_x[self.batch_idx] = instance.x()
        self.batch_y[self.batch_idx] = -1
        self.batch_idx += 1

        if self.batch_idx == self.batch_size:
            self.train_on_batch(self.batch_x, self.batch_y)
            self.batch_idx = 0
