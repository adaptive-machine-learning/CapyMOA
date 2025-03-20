import numpy as np


class _BatchBuilder:
    def __init__(
        self,
        batch_size: int,
        in_features: int,
        out_features: int,
        type_x: np.dtype,
        type_y: np.dtype,
    ):
        self.batch_x = np.ndarray((batch_size, in_features), dtype=type_x)
        self.batch_y = np.ndarray((batch_size, out_features), dtype=type_y)
        self.index = 0
        self.batch_size = batch_size

    def add(self, x: np.ndarray, y: np.ndarray) -> bool:
        """Add an element to the batch and if the batch is full return True

        The next call to add will overwrite the first element of the batch.

        :param x: An input feature vector
        :param y: An output feature vector
        :return: True if the batch is full, False otherwise
        """
        self.batch_x[self.index] = x
        self.batch_y[self.index] = y
        self.index += 1
        if self.index == self.batch_size:
            self.index = 0
            return True
        return False
