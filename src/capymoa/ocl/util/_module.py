from collections.abc import Mapping
from typing import Any

from torch import nn


class ModuleHolder(nn.Module):
    """Plain container that stores each keyword argument as a child module."""

    def __init__(self, **children: nn.Module) -> None:
        super().__init__()
        for name, child in children.items():
            self.add_module(name, child)


class HasModule:
    """Mixin for strategies that keep torch state in one private module.

    The class that uses this mixin must set ``self._module`` to an
    :class:`torch.nn.Module` in ``__init__``. The strategy itself is not an
    ``nn.Module``, so it exposes no module methods such as ``eval`` or ``cuda``.
    """

    _module: nn.Module

    @property
    def module(self) -> nn.Module:
        """Torch state of the strategy (``model`` plus strategy buffers).

        Use it to move the strategy to a device or to save and load a checkpoint.
        The model you passed in is at ``module.model``.
        """
        return self._module

    def state_dict(self) -> dict[str, Any]:
        """Return the state dict of :attr:`module`."""
        return self._module.state_dict()

    def load_state_dict(
        self, state_dict: Mapping[str, Any], strict: bool = True, assign: bool = False
    ):
        """Load a state dict into :attr:`module`.

        Values are copied in place, so existing optimisers stay valid.
        """
        return self._module.load_state_dict(state_dict, strict=strict, assign=assign)
