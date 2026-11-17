from itertools import chain
from typing import Generic, TypeVar

import torch
from torch import nn

M = TypeVar("M", bound=nn.Module)


class TorchMixin(Generic[M]):
    """Mixin for learners that keep their torch state in one private module.

    The class that uses this mixin must set ``self._module`` to its own
    :class:`torch.nn.Module` subclass ``M`` in ``__init__``. The learner itself is
    not an ``nn.Module``, so it has no module methods such as ``eval`` or ``cuda``.
    """

    _module: M

    @property
    def module(self) -> M:
        """Torch state of the learner (``model`` plus buffers).

        Use it to move the learner to a device (``learner.module.to(...)``) or to
        save and load a checkpoint, for example with
        ``learner.module.state_dict()`` and ``learner.module.load_state_dict(...)``.
        The model you passed in is at ``module.model``.
        """
        return self._module

    @property
    def device(self) -> torch.device:
        """Device of the first parameter or buffer of :attr:`module`.

        It is CPU if the module has none. It follows ``module.to(...)``.
        """
        for tensor in chain(self._module.parameters(), self._module.buffers()):
            return tensor.device
        return torch.device("cpu")
