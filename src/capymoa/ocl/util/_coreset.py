from typing import Tuple
from typing_extensions import override
from torch import Tensor
from abc import abstractmethod, ABC
import torch


class Coreset(ABC):
    @abstractmethod
    def update(self, x: Tensor, y: Tensor) -> None:
        """Update the coreset with new examples.

        :param x: Tensor of shape (batch, features)
        :param y: Tensor of shape (batch,) with class labels
        """
        ...

    def sample(self, n: int) -> Tuple[Tensor, Tensor]:
        """Sample ``n`` examples from the coreset.

        :param n: Number of examples to sample
        :return: Tuple of (x, y) where x is a Tensor of shape (n, features) and y is a
            Tensor of shape (n,) with class labels
        """
        indices = torch.randint(0, self.count, (n,))
        return self._buffer_x[indices], self._buffer_y[indices]

    def array(self) -> Tuple[Tensor, Tensor]:
        """Return the coreset as a tuple of (x, y) tensors."""
        return self._buffer_x[: self._count], self._buffer_y[: self._count]

    @property
    def capacity(self) -> int:
        """Return the maximum number of samples that can be stored in the coreset."""
        return self._capacity

    @property
    def count(self) -> int:
        """Return the current number of samples in the coreset."""
        assert self._count <= self._capacity
        return self._count

    @property
    def device(self) -> torch.device:
        return self._buffer_x.device

    def __init__(self, capacity: int, features: int, rng: torch.Generator):
        super().__init__()
        self._capacity = capacity
        self._features = features
        self._rng = rng
        self._count = 0
        self._buffer_x = torch.zeros((capacity, features))
        self._buffer_y = torch.zeros((capacity,), dtype=torch.long)


class ReservoirSampler(Coreset):
    def __init__(self, capacity: int, features: int, rng: torch.Generator):
        super().__init__(capacity, features, rng)
        self._i = 0

    @override
    def update(self, x: Tensor, y: Tensor) -> None:
        x = x.to(self.device)
        y = y.to(self.device)
        batch_size = x.shape[0]
        assert x.shape == (
            batch_size,
            self._features,
        )
        assert y.shape == (batch_size,)

        for i in range(batch_size):
            if self.count < self.capacity:
                # Fill the reservoir
                self._buffer_x[self.count] = x[i]
                self._buffer_y[self.count] = y[i]
                self._count += 1
            else:
                # Reservoir sampling
                index = torch.randint(0, self._i + 1, (1,), generator=self._rng)
                if index < self.capacity:
                    self._buffer_x[index] = x[i]
                    self._buffer_y[index] = y[i]
            self._i += 1


class GreedySampler(Coreset):
    """Update the buffer with every new example, replacing a random example from the
    majority class if the buffer is full.
    """

    @override
    def update(self, x: Tensor, y: Tensor) -> None:
        x = x.to(self.device)
        y = y.to(self.device)
        for xi, yi in zip(x, y):
            yi = int(yi.item())

            if self.count < self.capacity:
                # Room left in the coreset for this example
                self._buffer_x[self.count] = xi.cpu()
                self._buffer_y[self.count] = yi
                self._count += 1
            else:
                # Coreset is full, replace a random example from the majority class
                classes, counts = self._buffer_y.unique(return_counts=True)
                replace_class = classes[counts.argmax()].item()
                mask = self._buffer_y == replace_class
                idx = torch.randint(0, mask.sum(), (1,), generator=self._rng)
                replace_idx = mask.nonzero(as_tuple=True)[0][idx]
                self._buffer_x[replace_idx] = xi.cpu()
                self._buffer_y[replace_idx] = yi
