"""Resolve PyTorch model/optimizer classes from names, classes, or instances."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from torch import nn
    from torch.optim import Optimizer


def resolve_model(
    model: nn.Module | type[nn.Module] | str,
) -> nn.Module | type[nn.Module]:
    """Resolve ``model`` to an ``nn.Module`` instance or class.

    If ``model`` is a string, it is looked up by name in
    :mod:`capymoa.core.torch.ann`. Any other value is returned unchanged.

    :raises ValueError: If ``model`` is a string that does not name a known
        model in :mod:`capymoa.core.torch.ann`.
    """
    if isinstance(model, str):
        from capymoa.core.torch import ann

        try:
            return getattr(ann, model)
        except AttributeError:
            raise ValueError(
                f"Unknown model {model!r}. Known models: {ann.__all__}."
            ) from None
    return model


def resolve_optimizer(
    optimizer: Optimizer | type[Optimizer] | str,
) -> Optimizer | type[Optimizer]:
    """Resolve ``optimizer`` to an ``Optimizer`` instance or class.

    If ``optimizer`` is a string, it is looked up by name in
    :mod:`torch.optim`. Any other value is returned unchanged.

    :raises ValueError: If ``optimizer`` is a string that does not name a
        known optimizer in :mod:`torch.optim`.
    """
    if isinstance(optimizer, str):
        from torch import optim

        try:
            return getattr(optim, optimizer)
        except AttributeError:
            raise ValueError(f"Unknown optimizer {optimizer!r}.") from None
    return optimizer
