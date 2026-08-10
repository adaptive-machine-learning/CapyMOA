"""Artificial Neural Networks for CapyMOA."""

# PyTorch is an optional extra; this whole module requires it.
try:
    from ._perceptron import Perceptron
    from ._lenet import LeNet5
    from ._resnet import (
        resnet20_32x32,
        resnet32_32x32,
        resnet44_32x32,
        resnet56_32x32,
        resnet110_32x32,
        resnet1202_32x32,
    )
except ModuleNotFoundError as _err:  # pragma: no cover
    if (_err.name or "").split(".")[0] in ("torch", "torchvision"):
        from capymoa.exception import OptionalDependencyError

        raise OptionalDependencyError("PyTorch", "capymoa.core.torch.ann") from _err
    raise

__all__ = [
    "Perceptron",
    "LeNet5",
    "resnet20_32x32",
    "resnet32_32x32",
    "resnet44_32x32",
    "resnet56_32x32",
    "resnet110_32x32",
    "resnet1202_32x32",
]
