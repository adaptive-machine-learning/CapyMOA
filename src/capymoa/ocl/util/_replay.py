from abc import ABC, abstractmethod
from collections.abc import Mapping

import torch
from torch import Tensor, nn
from torch.utils.data import TensorDataset
from typing_extensions import override

#: Maps a buffer key to the shape (excluding batch) and dtype of its tensor.
BufferSpec = Mapping[str, tuple[tuple[int, ...], torch.dtype]]


class ReplayBuffer(ABC, nn.Module):
    @abstractmethod
    def update(self, **batch: Tensor) -> None:
        """Update the replay buffer with new examples.

        :param batch: Tensors with the same keys as the buffer's spec. Each has a
            leading batch dimension. Tensors are detached before storing.
        """
        ...

    def sample(self, n: int) -> dict[str, Tensor]:
        """Sample ``n`` examples (with replacement) from the replay buffer.

        :param n: Number of examples to sample
        :return: Dictionary with the same keys as the spec. Each tensor has ``n``
            rows.
        """
        indices = torch.randint(0, self.count, (n,), generator=self._rng)
        return {k: v[indices] for k, v in self._buffer.items()}

    def array(self) -> dict[str, Tensor]:
        """Return the stored examples as a dictionary of tensors."""
        return {k: v[: self._count] for k, v in self._buffer.items()}

    def dataset_view(self) -> TensorDataset:
        """Return a TensorDataset view of the replay buffer."""
        return TensorDataset(*self.array().values())

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
    def _buffer(self) -> dict[str, Tensor]:
        return {k: getattr(self, f"buffer_{k}") for k in self._spec}

    @property
    def device(self) -> torch.device:
        return next(iter(self._buffer.values())).device

    def __init__(
        self,
        capacity: int,
        spec: BufferSpec,
        rng: torch.Generator | None = None,
    ) -> None:
        """Construct a replay buffer.

        :param capacity: Maximum number of examples to store.
        :param spec: Shape (excluding batch) and dtype of each stored tensor, by key.
        :param rng: Random number generator, defaults to an unseeded generator.
        """
        super().__init__()
        if rng is None:
            rng = torch.Generator()
        self._capacity = capacity
        self._spec = dict(spec)
        self._rng = rng
        self._count = 0
        for key, (shape, dtype) in spec.items():
            self.register_buffer(
                f"buffer_{key}", torch.zeros((capacity, *shape), dtype=dtype)
            )
        self._i = 0

    def _prepare(self, batch: dict[str, Tensor]) -> tuple[dict[str, Tensor], int]:
        """Check keys and shapes, detach and move to the buffer device."""
        assert set(batch.keys()) == set(self._spec.keys())
        batch_size = next(iter(batch.values())).shape[0]
        out = {}
        for key, values in batch.items():
            shape, _ = self._spec[key]
            assert values.shape == (batch_size, *shape)
            out[key] = values.detach().to(self._buffer[key].device)
        return out, batch_size


class ReservoirSampler(ReplayBuffer):
    @override
    def update(self, **batch: Tensor) -> None:
        batch, batch_size = self._prepare(batch)
        for i in range(batch_size):
            if self.count < self.capacity:
                # Fill the reservoir
                for key, values in batch.items():
                    self._buffer[key][self.count] = values[i]
                self._count += 1
            else:
                # Reservoir sampling
                index = int(
                    torch.randint(0, self._i + 1, (1,), generator=self._rng).item()
                )
                if index < self.capacity:
                    for key, values in batch.items():
                        self._buffer[key][index] = values[i]
            self._i += 1


class GreedySampler(ReplayBuffer):
    """Update the buffer with every new example, replacing a random example from the
    majority class if the buffer is full.

    The spec must contain a ``y`` key with integer class labels.
    """

    @override
    def update(self, **batch: Tensor) -> None:
        batch, batch_size = self._prepare(batch)
        for i in range(batch_size):
            if self.count < self.capacity:
                # Room left in the coreset for this example
                for key, values in batch.items():
                    self._buffer[key][self.count] = values[i]
                self._count += 1
            else:
                # Coreset is full, replace a random example from the majority class
                y = self._buffer["y"]
                classes, counts = y.unique(return_counts=True)
                replace_class = classes[counts.argmax()].item()
                mask = y == replace_class
                idx = torch.randint(0, mask.sum(), (1,), generator=self._rng)
                replace_idx = mask.nonzero(as_tuple=True)[0][idx]
                for key, values in batch.items():
                    self._buffer[key][replace_idx] = values[i]


class SlidingWindow(ReplayBuffer):
    """Update the buffer with every new example, replacing the oldest example if the
    buffer is full.
    """

    @override
    def update(self, **batch: Tensor) -> None:
        batch, batch_size = self._prepare(batch)

        # Calculate where the batch ends
        end_idx = self._i + batch_size

        for key, values in batch.items():
            buf = self._buffer[key]
            if end_idx <= self.capacity:
                # Case 1: Simple slice (no wrap-around)
                buf[self._i : end_idx] = values
            else:
                # Case 2: Wrap-around (split the batch)
                mid_point = self.capacity - self._i
                buf[self._i :] = values[:mid_point]
                buf[: batch_size - mid_point] = values[mid_point:]

        # Update index and count
        self._i = (self._i + batch_size) % self.capacity
        self._count = min(self._count + batch_size, self.capacity)
