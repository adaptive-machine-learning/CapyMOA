"""PyTorch utilities for CapyMOA.

See :mod:`capymoa.core.torch.ann` for artificial neural network architectures.
"""

try:
    from ._interface import FeatureModel, ProjectionModel
except ModuleNotFoundError as _err:  # pragma: no cover
    if (_err.name or "").split(".")[0] in ("torch", "torchvision"):
        from capymoa.exception import OptionalDependencyError

        raise OptionalDependencyError("PyTorch", "capymoa.core.torch") from _err
    raise

__all__ = ["FeatureModel", "ProjectionModel"]
