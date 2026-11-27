from torch import Tensor, nn

from capymoa.core.torch import FeatureModel
from capymoa.stream._stream import Schema


class Perceptron(FeatureModel):
    """A simple feedforward neural network with one hidden layer."""

    def __init__(self, schema: Schema, hidden_size: int = 50):
        """Initialize the model.

        :param schema: Describes the stream's structure and types.
        :param hidden_size: Number of hidden units in the first layer.
        """
        super().__init__()
        in_features = schema.get_num_attributes()
        out_features = schema.get_num_classes()
        self._fc1 = nn.Linear(in_features, hidden_size)
        self._relu = nn.ReLU()
        self._fc2 = nn.Linear(hidden_size, out_features)

    def features(self, x: Tensor) -> Tensor:
        """Return the hidden representation of the input.

        :param x: Input tensor of shape ``(batch_size, num_features)``.
        :return: Hidden features of shape ``(batch_size, hidden_size)``.
        """
        x = x.view(x.size(0), -1)  # Flatten input
        x = self._fc1(x)
        return self._relu(x)

    def forward(self, x: Tensor) -> Tensor:
        """Return class logits for the input."""
        return self._fc2(self.features(x))
