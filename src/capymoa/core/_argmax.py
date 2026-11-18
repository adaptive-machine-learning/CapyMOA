import numpy as np

from ._instance import LabelProbabilities


def mersenne_twister(seed: int) -> np.random.Generator:
    """Mersenne Twister generator with the same seeding as Java.

    The seed goes through the original ``init_genrand``, so the raw 32-bit
    outputs match those of ``MersenneTwister(seed)`` in Java (commons-math).

    >>> from capymoa.core import mersenne_twister
    >>> int(mersenne_twister(5489).bit_generator.random_raw())
    3499211612

    :param seed: Seed for the generator.
    :return: A generator backed by ``MT19937``.
    """
    bit_generator = np.random.MT19937()
    bit_generator._legacy_seeding(seed)
    return np.random.Generator(bit_generator)


def random_argmax(x: np.ndarray, rng: np.random.Generator) -> int:
    """Index of the largest value, breaking ties at random.

    Unlike :func:`numpy.argmax`, which always picks the first of equal values,
    this picks one of the ``k`` tied indices. A tie uses one raw 32-bit output
    ``r`` of ``rng`` and picks tied index number ``(r * k) >> 32``. This is easy
    to repeat in Java, so Python and Java agree. No tie uses no output.

    >>> import numpy as np
    >>> from capymoa.core import mersenne_twister, random_argmax
    >>> rng = mersenne_twister(1)
    >>> random_argmax(np.array([0.1, 0.7, 0.2]), rng)
    1
    >>> sorted({random_argmax(np.zeros(3), rng) for _ in range(100)})
    [0, 1, 2]

    :param x: Values to search. Flattened first.
    :param rng: Generator from :func:`mersenne_twister`.
    :return: Index of a largest value.
    """
    x = np.asarray(x).ravel()
    tied = np.flatnonzero(x == x.max())
    if len(tied) == 1:
        return int(tied[0])
    bit_generator = rng.bit_generator
    if not isinstance(bit_generator, np.random.MT19937):
        raise TypeError("rng must be backed by MT19937, see mersenne_twister")
    raw = int(bit_generator.random_raw())
    return int(tied[(raw * len(tied)) >> 32])


def uniform_proba(num_classes: int) -> LabelProbabilities:
    """Uniform distribution over ``num_classes`` labels.

    >>> from capymoa.core import uniform_proba
    >>> uniform_proba(4)
    array([0.25, 0.25, 0.25, 0.25])

    :param num_classes: Number of labels.
    :return: Probabilities that sum to one.
    """
    return np.full(num_classes, 1.0 / num_classes, dtype=np.float64)
