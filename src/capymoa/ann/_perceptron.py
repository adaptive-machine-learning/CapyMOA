from capymoa.stream._stream import Schema
from torch import nn
from torch import Tensor


class Perceptron(nn.Module):
    """A simple feedforward neural network with one hidden layer."""

    def __init__(self, schema: Schema, hidden_size: int = 50):
        """Initialize the model.

        :param schema: Schema describing the data types and shapes.
        :param hidden_size: Number of hidden units in the first layer.
        """
        super(Perceptron, self).__init__()
        in_features = schema.get_num_attributes()
        out_features = schema.get_num_classes()
        self._fc1 = nn.Linear(in_features, hidden_size)
        self._relu = nn.ReLU()
        self._fc2 = nn.Linear(hidden_size, out_features)

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass through the network.

        :param x: Input tensor of shape ``(batch_size, num_features)``.
        :return: Output tensor of shape ``(batch_size, num_classes)``.
        """
        x = self._fc1(x)
        x = self._relu(x)
        x = self._fc2(x)
        return x
