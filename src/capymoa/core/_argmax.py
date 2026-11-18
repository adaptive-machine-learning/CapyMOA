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


def _next_int(rng: np.random.Generator, n: int) -> int:
    """Integer in ``[0, n)``, as ``nextInt(n)`` of Java's commons-math.

    Uses the top 31 bits of one raw 32-bit output. A power of two scales the
    bits. Any other ``n`` takes the remainder and redraws if the 31-bit value
    falls in the uneven top range.
    """
    bit_generator = rng.bit_generator
    if not isinstance(bit_generator, np.random.MT19937):
        raise TypeError("rng must be backed by MT19937, see mersenne_twister")
    if n & -n == n:
        return (n * (int(bit_generator.random_raw()) >> 1)) >> 31
    while True:
        bits = int(bit_generator.random_raw()) >> 1
        val = bits % n
        if bits - val + (n - 1) < 2**31:  # Java checks this for int overflow
            return val


def random_argmax(
    x: np.ndarray, rng: np.random.Generator, num_classes: int | None = None
) -> int:
    """Index of the largest value, breaking ties at random.

    Unlike :func:`numpy.argmax`, which always picks the first of equal values,
    this picks one of the ``k`` tied indices. It follows ``randMaxIndex`` in
    MOA's Java evaluation loops, so Python and Java agree:

    * ``NaN`` never wins.
    * No tie: return the index and do not use ``rng``.
    * A tie: draw ``nextInt(k)`` and return the tied index with that rank.
    * Empty or all ``NaN`` values: every label ties, so draw
      ``nextInt(num_classes)``. With fewer than two labels return 0.

    >>> import numpy as np
    >>> from capymoa.core import mersenne_twister, random_argmax
    >>> rng = mersenne_twister(1)
    >>> random_argmax(np.array([0.1, 0.7, 0.2]), rng)
    1
    >>> sorted({random_argmax(np.zeros(3), rng) for _ in range(100)})
    [0, 1, 2]

    :param x: Values to search. Flattened first.
    :param rng: Generator from :func:`mersenne_twister`.
    :param num_classes: Number of labels. Values past it are ignored, and it
        is the number of ties when ``x`` has no valid value. Defaults to
        ``len(x)``.
    :return: Index of a largest value.
    """
    x = np.asarray(x, dtype=np.float64).ravel()
    if num_classes is not None:
        # Some learners return more votes than the stream has labels.
        x = x[:num_classes]
    valid = ~np.isnan(x)
    if not valid.any():
        n = len(x) if num_classes is None else num_classes
        return 0 if n < 2 else _next_int(rng, n)
    tied = np.flatnonzero(x == x[valid].max())
    if len(tied) == 1:
        return int(tied[0])
    return int(tied[_next_int(rng, len(tied))])


def uniform_proba(num_classes: int) -> LabelProbabilities:
    """Uniform distribution over ``num_classes`` labels.

    >>> from capymoa.core import uniform_proba
    >>> uniform_proba(4)
    array([0.25, 0.25, 0.25, 0.25])

    :param num_classes: Number of labels.
    :return: Probabilities that sum to one.
    """
    return np.full(num_classes, 1.0 / num_classes, dtype=np.float64)
