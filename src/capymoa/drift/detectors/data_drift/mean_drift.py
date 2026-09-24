"""Mean-shift drift detector for data drift."""

from typing import Any

import numpy as np

from .base import BaseDataDriftDetector, DataDriftResult


class MeanDriftDetector(BaseDataDriftDetector):
    """Mean-Shift Drift Detector

    Detects drift by computing the mean squared deviation (MSD) between
    the feature-wise means of the reference and test windows. Detection
    thresholds are derived from permutations of the reference data: the
    reference is repeatedly split into a reference-sized part and a
    test-sized part to establish what the MSD looks like under no drift,
    for the actual reference and test window sizes in use.

    Drift is declared when the observed MSD exceeds
    ``mean + 2 * std`` of the permutation-based baseline.

    This detector is inspired by the Image-Based Drift Detector (IBDD) of
    Souza et al. (2021), which compares reference and test windows as
    images. It does **not** implement that algorithm: it does not render
    windows as images, does not require consecutive threshold crossings,
    and does not adapt separate upper/lower thresholds. It only compares
    feature-wise means, calibrated by permutation.

    Example:
    --------

    >>> import numpy as np
    >>> from capymoa.drift.detectors import MeanDriftDetector
    >>> rng = np.random.default_rng(42)
    >>> detector = MeanDriftDetector(window_size=50, n_permutations=50)
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
        seed: int | None = None,
        auto_fit_samples: int | None = None,
    ):
        """Create a mean-shift data drift detector.

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
            window_size,
            alpha=alpha,
            correction="none",
            auto_fit_samples=auto_fit_samples,
        )
        self._n_permutations = n_permutations
        self._seed = seed
        self._threshold_cache: dict[int, float] = {}

    def _fit(self, X: np.ndarray) -> None:
        self._X_ref = X
        self._threshold_cache = {}

    def _get_threshold(self, test_size: int) -> float:
        """Return the MSD threshold calibrated for *test_size*, caching it."""
        if test_size not in self._threshold_cache:
            self._threshold_cache[test_size] = self._compute_threshold(
                self._X_ref, test_size
            )
        return self._threshold_cache[test_size]

    def _compute_threshold(self, X: np.ndarray, test_size: int) -> float:
        """Establish the MSD threshold from permutations of *X*.

        Each permutation draws a *test_size*-sized part and compares its
        mean against the mean of the remaining part, matching the shape
        of the actual reference-vs-test comparison in :meth:`_test`.
        """
        n = len(X)
        if test_size >= n:
            raise ValueError(
                f"test size ({test_size}) must be smaller than the "
                f"reference size ({n}) to calibrate a threshold"
            )
        rng = np.random.default_rng(self._seed)
        msds = np.empty(self._n_permutations)
        for i in range(self._n_permutations):
            perm = rng.permutation(n)
            test_idx = perm[:test_size]
            ref_idx = perm[test_size:]
            mean_test = X[test_idx].mean(axis=0)
            mean_ref = X[ref_idx].mean(axis=0)
            msds[i] = float(np.mean((mean_ref - mean_test) ** 2))
        return float(msds.mean() + 2.0 * msds.std())

    def _test(self, X_ref: np.ndarray, X_test: np.ndarray) -> DataDriftResult:
        threshold = self._get_threshold(X_test.shape[0])
        msd = float(np.mean((X_ref.mean(axis=0) - X_test.mean(axis=0)) ** 2))
        return DataDriftResult(
            is_drift=msd > threshold,
            statistic=msd,
            distance=msd,
        )

    def get_params(self) -> dict[str, Any]:
        return {
            "window_size": self._window_size,
            "alpha": self._alpha,
            "n_permutations": self._n_permutations,
            "seed": self._seed,
            "auto_fit_samples": self._auto_fit_samples,
        }
