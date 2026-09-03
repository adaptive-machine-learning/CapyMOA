"""Image-Based Drift Detector (IBDD) for data drift."""

from typing import Any, Dict, Optional

import numpy as np

from capymoa.drift.detectors.data_drift.base import BaseDataDriftDetector, DataDriftResult


class IBDD(BaseDataDriftDetector):
    """Image-Based Drift Detector

    Detects drift by computing the mean squared deviation (MSD) between
    the feature-wise means of the reference and test windows. Detection
    thresholds are derived from permutations of the reference data: the
    reference is shuffled and split repeatedly to establish what the MSD
    looks like under no drift.

    Drift is declared when the observed MSD exceeds
    ``mean + 2 * std`` of the permutation-based baseline.

    Example:
    --------

    >>> import numpy as np
    >>> from capymoa.drift.detectors.data_drift import IBDD
    >>> rng = np.random.default_rng(42)
    >>> detector = IBDD(window_size=50, n_permutations=50)
    >>> detector.fit(rng.normal(0, 1, size=(200, 2)))
    >>> for x in rng.normal(3, 1, size=(50, 2)):
    ...     detector.add_element(x)
    >>> detector.detected_change()
    True

    Reference:
    ----------

    Souza, V. M. A., Parmezan, A. R. S., Chowdhury, F. A., and Mueen, A.
    "Efficient unsupervised drift detector for fast and high-dimensional data
    streams." Knowledge and Information Systems (2021). Springer.

    """

    IS_UNIVARIATE = False

    def __init__(
        self,
        window_size: int,
        alpha: float = 0.05,
        n_permutations: int = 50,
        seed: Optional[int] = None,
        auto_fit_samples: Optional[int] = None,
    ):
        """Create an IBDD data drift detector.

        :param window_size: Number of observations in the sliding window.
        :param alpha: Not used for threshold (permutation-based), kept
            for interface compatibility.
        :param n_permutations: Number of reference permutations used to
            establish the baseline MSD distribution. More permutations
            give a more stable threshold.
        :param seed: Random seed for the permutation step.
        :param auto_fit_samples: Number of initial samples for auto-fit.
        :raises ValueError: If *n_permutations* < 1.
        """
        if n_permutations < 1:
            raise ValueError("n_permutations must be at least 1")
        super().__init__(
            window_size, alpha=alpha, correction="none",
            auto_fit_samples=auto_fit_samples,
        )
        self._n_permutations = n_permutations
        self._seed = seed
        self._upper_threshold: Optional[float] = None

    def _fit(self, X: np.ndarray) -> None:
        self._X_ref = X
        self._ref_mean = X.mean(axis=0)
        self._compute_threshold(X)

    def _compute_threshold(self, X: np.ndarray) -> None:
        """Establish the MSD threshold from permutations of *X*.

        The reference is split into two random halves repeatedly. The
        MSD between their means approximates the null distribution.
        """
        rng = np.random.default_rng(self._seed)
        n = len(X)
        half = n // 2
        msds = np.empty(self._n_permutations)
        for i in range(self._n_permutations):
            perm = rng.permutation(n)
            mean_a = X[perm[:half]].mean(axis=0)
            mean_b = X[perm[half : 2 * half]].mean(axis=0)
            msds[i] = float(np.mean((mean_a - mean_b) ** 2))
        self._upper_threshold = float(msds.mean() + 2.0 * msds.std())

    def _test(self, X_ref: np.ndarray, X_test: np.ndarray) -> DataDriftResult:
        msd = float(np.mean((X_ref.mean(axis=0) - X_test.mean(axis=0)) ** 2))
        return DataDriftResult(
            is_drift=msd > self._upper_threshold,
            statistic=msd,
            distance=msd,
        )

    def get_params(self) -> Dict[str, Any]:
        return {
            "window_size": self._window_size,
            "n_permutations": self._n_permutations,
            "seed": self._seed,
        }
