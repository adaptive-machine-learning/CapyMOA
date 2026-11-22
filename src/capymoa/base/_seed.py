import numpy as np


def resolve_seed(random_seed: int | None) -> int:
    """Return ``random_seed``, or draw a new seed if ``None``."""
    if random_seed is None:
        return np.random.randint(np.iinfo(np.int32).max)
    return random_seed


class _SeededMixin:
    """Mixin defining the random-seed behavior for learners."""

    _random_seed: int

    def _init_seed(self, random_seed: int | None) -> None:
        """Initialize the random seed for the learner."""
        self._random_seed = resolve_seed(random_seed)

    @property
    def random_seed(self) -> int:
        """Random seed for reproducibility."""
        return self._random_seed
