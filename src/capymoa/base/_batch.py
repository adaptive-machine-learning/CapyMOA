import torch
from abc import ABC, abstractmethod


class Batch(ABC):
    """Base class for batch processing in CapyMOA"""

    x_dtype: torch.dtype
    """Data type for the input features."""
    y_dtype: torch.dtype
    """Data type for the target value/labels."""
    device: torch.device = torch.device("cpu")
    """Device on which the batch will be processed."""

    @abstractmethod
    def batch_train(self, x: torch.Tensor, y: torch.Tensor) -> None:
        """Train the model with a batch of instances.

        :param x: A batch of feature vectors of shape ``(batch_size, num_features)``.
        :param y: A batch of target values, typically a vector of shape ``(batch_size,)``.
        """

    @abstractmethod
    def batch_predict(self, x: torch.Tensor) -> torch.Tensor:
        """Predict the target values for a batch of instances.

        :param x: A batch of feature vectors of shape ``(batch_size, num_features)``.
        :return: Predicted target values, typically a vector of shape ``(batch_size,)``.
        """
