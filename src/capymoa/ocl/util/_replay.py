from abc import ABC, abstractmethod

import torch
from torch import Tensor, nn
from torch.utils.data import TensorDataset
from typing_extensions import override


class ReplayBuffer(ABC, nn.Module):
    @abstractmethod
    def update(self, x: Tensor, y: Tensor) -> None:
        """Update the replay buffer with new examples.

        :param x: Tensor of shape (batch, features)
        :param y: Tensor of shape (batch,) with class labels
        """
        ...

    def sample(self, n: int) -> tuple[Tensor, Tensor]:
        """Sample ``n`` examples from the replay buffer.

        :param n: Number of examples to sample
        :return: Tuple of (x, y) where x is a Tensor of shape (n, features) and y is a
            Tensor of shape (n,) with class labels
        """
        indices = torch.randint(0, self.count, (n,), generator=self._rng)
        return self._buffer_x[indices], self._buffer_y[indices]

    def array(self) -> tuple[Tensor, Tensor]:
        """Return the replay buffer as a tuple of (x, y) tensors."""
        return self._buffer_x[: self._count], self._buffer_y[: self._count]

    def dataset_view(self) -> TensorDataset:
        """Return a TensorDataset view of the replay buffer."""
        return TensorDataset(*self.array())

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

    def __init__(
        self,
        capacity: int,
        features: int,
        rng: torch.Generator | None = None,
    ) -> None:
        super().__init__()
        if rng is None:
            rng = torch.Generator()
        self._capacity = capacity
        self._features = features
        self._rng = rng
        self._count = 0
        self._buffer_x = nn.Buffer(torch.zeros((capacity, features)))
        self._buffer_y = nn.Buffer(torch.zeros((capacity,), dtype=torch.long))
        self._i = 0


class ReservoirSampler(ReplayBuffer):
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


class GreedySampler(ReplayBuffer):
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


class SlidingWindow(ReplayBuffer):
    """Update the buffer with every new example, replacing the oldest example if the
    buffer is full.
    """

    @override
    def update(self, x: Tensor, y: Tensor) -> None:
        x = x.to(self.device)
        y = y.to(self.device)
        batch_size = x.shape[0]

        # Calculate where the batch ends
        end_idx = self._i + batch_size

        if end_idx <= self.capacity:
            # Case 1: Simple slice (no wrap-around)
            self._buffer_x[self._i : end_idx] = x
            self._buffer_y[self._i : end_idx] = y
        else:
            # Case 2: Wrap-around (split the batch)
            mid_point = self.capacity - self._i

            # Fill the end of the buffer
            self._buffer_x[self._i :] = x[:mid_point]
            self._buffer_y[self._i :] = y[:mid_point]

            # Wrap the remainder to the start
            wrap_size = batch_size - mid_point
            self._buffer_x[:wrap_size] = x[mid_point:]
            self._buffer_y[:wrap_size] = y[mid_point:]

        # Update index and count
        self._i = (self._i + batch_size) % self.capacity
        self._count = min(self._count + batch_size, self.capacity)
