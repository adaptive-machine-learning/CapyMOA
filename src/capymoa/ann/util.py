from torch import nn
from typing import cast


def apply_weight_norm(linear: nn.Linear):
    """Apply weight normalization to a linear layer.

    The output layer of a neural network is often problematic in continual
    learning because of the extreme and shifting class imbalance between tasks.
    [Lesort2021]_ suggest mitigating this by using a variant of weight
    normalization that parameterize the weights as a magnitude (set to the unit
    vector) and a direction.

    .. [Lesort2021] Lesort, T., George, T., & Rish, I. (2021). Continual
       Learning in Deep Networks: An Analysis of the Last Layer.
    """
    linear = cast(
        nn.Linear, nn.utils.parametrizations.weight_norm(linear, name="weight")
    )
    linear.bias = None
    weight_g = linear.parametrizations.weight.original0
    # Set the magnitude to the unit vector
    weight_g.requires_grad_(False).fill_(1.0 / (linear.out_features**0.5))
    return linear
