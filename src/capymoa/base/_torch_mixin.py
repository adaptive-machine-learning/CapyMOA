from itertools import chain
from typing import Generic, TypeVar

import torch
from torch import nn

M = TypeVar("M", bound=nn.Module)


class TorchMixin(Generic[M]):
    """Mixin adding a torch module to a CapyMOA learner.

    This avoids classes inheriting directly from ``nn.Module`` while still
    providing a common API to the underlying torch module and its device.
    """

    _module: M

    @property
    def module(self) -> M:
        """The underlying torch module."""
        return self._module

    @property
    def device(self) -> torch.device:
        """Device of the first parameter or buffer of :attr:`module`."""
        for tensor in chain(self._module.parameters(), self._module.buffers()):
            return tensor.device
        return torch.device("cpu")
