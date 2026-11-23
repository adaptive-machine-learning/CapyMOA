"""PyTorch utilities for CapyMOA.

See :mod:`capymoa.core.torch.ann` for artificial neural network architectures.
"""

__all__ = ["FeatureModel", "ProjectionModel"]


def __getattr__(name: str):
    # ``capymoa.core`` imports this package, and PyTorch is an optional extra.
    # Load the interfaces on first use so ``import capymoa`` works without it.
    if name in __all__:
        try:
            from . import _interface
        except ModuleNotFoundError as err:  # pragma: no cover
            if (err.name or "").split(".")[0] in ("torch", "torchvision"):
                from capymoa.exception import OptionalDependencyError

                raise OptionalDependencyError("PyTorch", "capymoa.core.torch") from err
            raise
        return getattr(_interface, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
