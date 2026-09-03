"""Chi-square test for data drift on categorical features."""

import collections
from typing import Any, Dict, Literal, Optional

import numpy as np
from scipy.stats import chi2_contingency

from capymoa.drift.detectors.data_drift.base import BaseDataDriftDetector, DataDriftResult


class ChiSquare(BaseDataDriftDetector):
    """Chi-Square Test

    Tests whether the frequency distribution of categorical values has
    changed between reference and test samples. A contingency table is
    built from category counts and compared with
    :func:`scipy.stats.chi2_contingency`.

    This is the only data drift detector designed for **categorical**
    features. For numeric features use :class:`KolmogorovSmirnov` or
    other detectors.

    Applied per feature, with Bonferroni correction by default.

    Example:
    --------

    >>> import numpy as np
    >>> from capymoa.drift.detectors.data_drift import ChiSquare
    >>> rng = np.random.default_rng(42)
    >>> ref = rng.choice(["a", "b", "c"], size=(200, 2), p=[0.5, 0.3, 0.2])
    >>> detector = ChiSquare(window_size=50)
    >>> detector.fit(ref)
    >>> for x in rng.choice(["a", "b", "c"], size=(50, 2), p=[0.1, 0.3, 0.6]):
    ...     detector.add_element(x)
    >>> detector.detected_change()
    True

    Reference:
    ----------

    Pearson, Karl. "X. On the criterion that a given system of deviations from
    the probable in the case of a correlated system of variables is such that
    it can be reasonably supposed to have arisen from random sampling."
    The London, Edinburgh, and Dublin Philosophical Magazine and Journal of
    Science 50.302 (1900): 157-175.

    """

    IS_UNIVARIATE = True

    def __init__(
        self,
        window_size: int,
        alpha: float = 0.05,
        correction: Literal["bonferroni", "none"] = "bonferroni",
        auto_fit_samples: Optional[int] = None,
    ):
        """Create a Chi-square data drift detector.

        :param window_size: Number of observations in the sliding window.
        :param alpha: Significance level for drift decisions.
        :param correction: Multiple-testing correction across features.
        :param auto_fit_samples: Number of initial samples for auto-fit.
        """
        super().__init__(window_size, alpha, correction, auto_fit_samples)

    def _fit(self, X: np.ndarray) -> None:
        self._X_ref = X

    def _test(self, x_ref: np.ndarray, x_test: np.ndarray) -> DataDriftResult:
        ref_counts = collections.Counter(x_ref)
        test_counts = collections.Counter(x_test)

        all_categories = sorted(set(ref_counts) | set(test_counts))
        f_ref = [ref_counts.get(c, 0) for c in all_categories]
        f_test = [test_counts.get(c, 0) for c in all_categories]

        stat, p, _, _ = chi2_contingency(np.array([f_ref, f_test]))
        return DataDriftResult(is_drift=False, statistic=stat, p_value=p)

    def get_params(self) -> Dict[str, Any]:
        return {
            "window_size": self._window_size,
            "alpha": self._alpha,
            "correction": self._correction,
        }
