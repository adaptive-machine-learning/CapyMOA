"""Jensen-Shannon distance for data drift."""

from typing import Any, Literal

import numpy as np
from scipy.spatial.distance import jensenshannon

from .base import BaseDataDriftDetector, DataDriftResult, _bin_probabilities


class JensenShannon(BaseDataDriftDetector):
    """Jensen-Shannon Distance

    Computes the Jensen-Shannon distance (the square root of the
    Jensen-Shannon divergence) between reference and test distributions.
    Unlike KL divergence, JS distance is symmetric and bounded in
    ``[0, 1]``.

    Drift is declared when the distance exceeds ``threshold``.
    Applied independently to each feature using the configured
    threshold. Overall drift is reported when any feature flags drift.
    This detector does not return p-values, and the ``correction``
    parameter has no effect on its drift decisions. No Bonferroni
    correction is applied.

    Example:
    --------

    >>> import numpy as np
    >>> from capymoa.drift.detectors import JensenShannon
    >>> rng = np.random.default_rng(42)
    >>> detector = JensenShannon(window_size=50, num_bins=20, threshold=0.1)
    >>> detector.fit(rng.normal(0, 1, size=(200, 2)))
    >>> for x in rng.normal(3, 1, size=(50, 2)):
    ...     detector.add_element(x)
    >>> detector.detected_change()
    True

    Reference:
    ----------

    Lin, Jianhua. "Divergence measures based on the Shannon entropy."
    IEEE Transactions on Information Theory 37.1 (1991): 145-151.

    """

    IS_UNIVARIATE = True

    def __init__(
        self,
        window_size: int,
        num_bins: int = 10,
        threshold: float = 0.1,
        correction: Literal["bonferroni", "none"] = "bonferroni",
        auto_fit_samples: int | None = None,
    ):
        """Create a Jensen-Shannon data drift detector.

        :param window_size: Number of observations in the sliding window.
        :param num_bins: Number of histogram bins for probability
            estimation.
        :param threshold: Distance above which drift is declared
            (per feature). Must be in ``(0, 1]``.
        :param correction: Accepted for interface consistency with other
            detectors, but has no effect: this detector does not produce
            p-values, so no multiple-testing correction is applied to its
            drift decisions.
        :param auto_fit_samples: Number of initial samples for auto-fit.
        :raises ValueError: If *num_bins* < 1 or *threshold* not in
            ``(0, 1]``.
        """
        if num_bins < 1:
            raise ValueError("num_bins must be at least 1")
        if not 0.0 < threshold <= 1.0:
            raise ValueError("threshold must be in (0, 1]")
        super().__init__(
            window_size,
            alpha=0.05,
            correction=correction,
            auto_fit_samples=auto_fit_samples,
        )
        self._num_bins = num_bins
        self._threshold = threshold

    def _fit(self, X: np.ndarray) -> None:
        self._X_ref = X

    def _test(self, x_ref: np.ndarray, x_test: np.ndarray) -> DataDriftResult:
        ref_probs, test_probs = _bin_probabilities(x_ref, x_test, self._num_bins)
        js_dist = float(jensenshannon(ref_probs, test_probs))
        return DataDriftResult(
            is_drift=js_dist > self._threshold,
            statistic=js_dist,
            distance=js_dist,
        )

    def get_params(self) -> dict[str, Any]:
        return {
            "window_size": self._window_size,
            "num_bins": self._num_bins,
            "threshold": self._threshold,
            "correction": self._correction,
            "auto_fit_samples": self._auto_fit_samples,
        }
