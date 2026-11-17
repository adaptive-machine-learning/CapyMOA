from torch import nn


class TorchMixin:
    """Mixin for strategies that keep their torch state in one private module.

    The class that uses this mixin must set ``self._module`` to an
    :class:`torch.nn.Module` in ``__init__``. The strategy itself is not an
    ``nn.Module``, so it has no module methods such as ``eval`` or ``cuda``.
    """

    _module: nn.Module

    @property
    def module(self) -> nn.Module:
        """Torch state of the strategy (``model`` plus strategy buffers).

        Use it to move the strategy to a device or to save and load a checkpoint,
        for example with ``strategy.module.state_dict()`` and
        ``strategy.module.load_state_dict(...)``. The model you passed in is at
        ``module.model``.
        """
        return self._module
