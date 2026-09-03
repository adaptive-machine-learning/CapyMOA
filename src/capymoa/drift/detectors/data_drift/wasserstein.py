"""Wasserstein distance (Earth Mover's Distance) for data drift."""

from typing import Any, Dict, Literal, Optional

import numpy as np
from scipy.stats import wasserstein_distance

from capymoa.drift.detectors.data_drift.base import BaseDataDriftDetector, DataDriftResult


class Wasserstein(BaseDataDriftDetector):
    """Wasserstein Distance (Earth Mover's Distance)

    The first Wasserstein distance (EMD) measures the minimum "work"
    needed to transform one distribution into another. Intuitively, if
    the distributions are piles of earth, it is the minimum cost of
    moving earth to reshape one pile into the other.

    Drift is declared when the distance exceeds ``threshold``.
    Applied per feature, with Bonferroni correction by default.

    .. note::
        The distance is scale-dependent. Choose ``threshold`` based on
        the expected range of your features.

    Example:
    --------

    >>> import numpy as np
    >>> from capymoa.drift.detectors.data_drift import Wasserstein
    >>> rng = np.random.default_rng(42)
    >>> detector = Wasserstein(window_size=50, threshold=0.5)
    >>> detector.fit(rng.normal(0, 1, size=(200, 2)))
    >>> for x in rng.normal(3, 1, size=(50, 2)):
    ...     detector.add_element(x)
    >>> detector.detected_change()
    True

    Reference:
    ----------

    Rubner, Yossi, Carlo Tomasi, and Leonidas J. Guibas. "The earth mover's
    distance as a metric for image retrieval." International Journal of
    Computer Vision 40.2 (2000): 99-121.

    """

    IS_UNIVARIATE = True

    def __init__(
        self,
        window_size: int,
        threshold: float = 0.5,
        correction: Literal["bonferroni", "none"] = "bonferroni",
        auto_fit_samples: Optional[int] = None,
    ):
        """Create a Wasserstein data drift detector.

        :param window_size: Number of observations in the sliding window.
        :param threshold: Distance above which drift is declared
            (per feature).
        :param correction: Multiple-testing correction across features.
        :param auto_fit_samples: Number of initial samples for auto-fit.
        :raises ValueError: If *threshold* <= 0.
        """
        if threshold <= 0:
            raise ValueError("threshold must be positive")
        super().__init__(
            window_size, alpha=0.05, correction=correction,
            auto_fit_samples=auto_fit_samples,
        )
        self._threshold = threshold

    def _fit(self, X: np.ndarray) -> None:
        self._X_ref = X

    def _test(self, x_ref: np.ndarray, x_test: np.ndarray) -> DataDriftResult:
        dist = float(wasserstein_distance(x_ref, x_test))
        return DataDriftResult(
            is_drift=dist > self._threshold,
            statistic=dist,
            distance=dist,
        )

    def get_params(self) -> Dict[str, Any]:
        return {
            "window_size": self._window_size,
            "threshold": self._threshold,
            "correction": self._correction,
        }
