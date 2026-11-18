import numpy as np

from ._instance import LabelProbabilities


def random_argmax(x: np.ndarray, rng: np.random.Generator) -> int:
    """Index of the largest value, breaking ties at random.

    Unlike :func:`numpy.argmax`, which always picks the first of equal values,
    this picks one of the tied indices with ``rng``.

    >>> import numpy as np
    >>> from capymoa.core import random_argmax
    >>> rng = np.random.default_rng(1)
    >>> random_argmax(np.array([0.1, 0.7, 0.2]), rng)
    1
    >>> sorted({random_argmax(np.zeros(3), rng) for _ in range(100)})
    [0, 1, 2]

    :param x: Values to search. Flattened first.
    :param rng: Random generator used to break ties.
    :return: Index of a largest value.
    """
    x = np.asarray(x).ravel()
    return int(rng.choice(np.flatnonzero(x == x.max())))


def uniform_proba(num_classes: int) -> LabelProbabilities:
    """Uniform distribution over ``num_classes`` labels.

    >>> from capymoa.core import uniform_proba
    >>> uniform_proba(4)
    array([0.25, 0.25, 0.25, 0.25])

    :param num_classes: Number of labels.
    :return: Probabilities that sum to one.
    """
    return np.full(num_classes, 1.0 / num_classes, dtype=np.float64)
