"""Interfaces for PyTorch models that expose latent features."""

from abc import ABC, abstractmethod

# PyTorch is an optional extra; this whole module requires it.
from torch import Tensor, nn


class FeatureModel(nn.Module, ABC):
    """A classifier that exposes its latent features.

    Strategies that compare examples in latent space (for example ASER) need
    the features. The model defines them, so strategies do not guess.

    >>> import torch
    >>> class Net(FeatureModel):
    ...     def __init__(self):
    ...         super().__init__()
    ...         self.encoder = nn.Linear(2, 4)
    ...         self.head = nn.Linear(4, 3)
    ...     def features(self, x):
    ...         return self.encoder(x).relu()
    ...     def forward(self, x):
    ...         return self.head(self.features(x))
    >>> Net().features(torch.ones(5, 2)).shape
    torch.Size([5, 4])
    """

    @abstractmethod
    def features(self, x: Tensor) -> Tensor:
        """Return latent features with shape ``(batch, ...)``."""


class ProjectionModel(FeatureModel):
    """A :class:`FeatureModel` with a projection head for contrastive losses.

    The head is for training only. ``forward`` is not used by SCR.
    """

    @abstractmethod
    def project(self, features: Tensor) -> Tensor:
        """Map latent features to the space where the loss is computed."""
