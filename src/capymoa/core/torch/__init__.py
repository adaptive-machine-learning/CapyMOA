"""PyTorch utilities for CapyMOA.

See :mod:`capymoa.core.torch.ann` for artificial neural network architectures.
"""

from ._resolve import resolve_model, resolve_optimizer

__all__ = ["resolve_model", "resolve_optimizer"]
