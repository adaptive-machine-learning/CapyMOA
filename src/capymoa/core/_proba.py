import numpy as np

from ._instance import LabelProbabilities


def uniform_proba(num_classes: int) -> LabelProbabilities:
    """Uniform distribution over ``num_classes`` labels.

    >>> from capymoa.core import uniform_proba
    >>> uniform_proba(4)
    array([0.25, 0.25, 0.25, 0.25])

    :param num_classes: Number of labels.
    :return: Probabilities that sum to one.
    """
    return np.full(num_classes, 1.0 / num_classes, dtype=np.float64)
