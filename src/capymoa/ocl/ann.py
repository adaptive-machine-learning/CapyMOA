import torch
from torch import Tensor, nn

from capymoa.stream import Schema


class WNPerceptron(nn.Module):
    """A simple one hidden layer feedforward neural network with

    The output layer of a neural network is often problematic in continual
    learning because of the extreme and shifting class imbalance between tasks.
    [Lesort2021]_ suggest mitigating this by using a variant of weight
    normalization that parameterize the weights as a magnitude (set to the unit
    vector) and a direction.

    .. [Lesort2021] Lesort, T., George, T., & Rish, I. (2021). Continual
       Learning in Deep Networks: An Analysis of the Last Layer.
    """

    def __init__(self, schema: Schema, hidden_size: int = 50):
        super().__init__()
        num_classes = schema.get_num_classes()

        self.fc1 = nn.Linear(schema.get_num_attributes(), hidden_size)
        self.fc2 = nn.Linear(hidden_size, num_classes, bias=False)
        self.fc2 = nn.utils.parametrizations.weight_norm(self.fc2, name="weight")
        weight_g = self.fc2.parametrizations.weight.original0
        # Set the magnitude to the unit vector
        weight_g.requires_grad_(False).fill_(1.0 / (num_classes**0.5))

    def forward(self, x: Tensor) -> Tensor:
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x
