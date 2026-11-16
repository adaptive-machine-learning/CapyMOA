import numpy as np


def resolve_seed(random_seed: int | None) -> int:
    """Return ``random_seed``, or a seed from ``numpy.random.default_rng()`` if ``None``."""
    if random_seed is None:
        return int(np.random.default_rng().integers(2**31))
    return random_seed


class SeededMixin:
    """Gives a learner a read-only ``random_seed``.

    Call :meth:`_init_seed` in the constructor. A ``None`` seed is replaced by
    one from ``numpy.random.default_rng()``, so the seed used is always known.
    """

    _random_seed: int

    def _init_seed(self, random_seed: int | None) -> None:
        self._random_seed = resolve_seed(random_seed)

    @property
    def random_seed(self) -> int:
        """The random seed used (read-only)."""
        return self._random_seed
