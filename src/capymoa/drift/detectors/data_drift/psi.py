"""Population Stability Index (PSI) for data drift."""

import sys
from typing import Any, Dict, Literal, Optional

import numpy as np

from capymoa.drift.detectors.data_drift.base import BaseDataDriftDetector, DataDriftResult


class PSI(BaseDataDriftDetector):
    """Population Stability Index

    PSI measures the shift between reference and test distributions by
    comparing binned proportions. It is widely used in credit risk
    modelling to monitor score distributions over time.

    Standard interpretation:

    - PSI < 0.1 -- no significant change.
    - 0.1 <= PSI < 0.2 -- moderate change.
    - PSI >= 0.2 -- significant change.

    Drift is declared when PSI exceeds ``threshold`` (default 0.2).
    Applied per feature, with Bonferroni correction by default.

    Example:
    --------

    >>> import numpy as np
    >>> from capymoa.drift.detectors.data_drift import PSI
    >>> rng = np.random.default_rng(42)
    >>> detector = PSI(window_size=50, num_bins=20)
    >>> detector.fit(rng.normal(0, 1, size=(200, 2)))
    >>> for x in rng.normal(3, 1, size=(50, 2)):
    ...     detector.add_element(x)
    >>> detector.detected_change()
    True

    Reference:
    ----------

    Wu, Desheng, and David L. Olson. "Enterprise risk management: coping with
    model risk in a large bank." Journal of the Operational Research Society
    61.2 (2010): 179-190.

    """

    IS_UNIVARIATE = True

    def __init__(
        self,
        window_size: int,
        num_bins: int = 10,
        threshold: float = 0.2,
        correction: Literal["bonferroni", "none"] = "bonferroni",
        auto_fit_samples: Optional[int] = None,
    ):
        """Create a PSI data drift detector.

        :param window_size: Number of observations in the sliding window.
        :param num_bins: Number of equal-width bins for proportion
            estimation.
        :param threshold: PSI above which drift is declared (per
            feature). Default is 0.2 (significant change).
        :param correction: Multiple-testing correction across features.
        :param auto_fit_samples: Number of initial samples for auto-fit.
        :raises ValueError: If *num_bins* < 1 or *threshold* <= 0.
        """
        if num_bins < 1:
            raise ValueError("num_bins must be at least 1")
        if threshold <= 0:
            raise ValueError("threshold must be positive")
        super().__init__(
            window_size, alpha=0.05, correction=correction,
            auto_fit_samples=auto_fit_samples,
        )
        self._num_bins = num_bins
        self._threshold = threshold

    def _fit(self, X: np.ndarray) -> None:
        self._X_ref = X

    def _test(self, x_ref: np.ndarray, x_test: np.ndarray) -> DataDriftResult:
        combined = np.concatenate([x_ref, x_test])
        edges = np.linspace(combined.min(), combined.max(), self._num_bins + 1)
        ref_counts, _ = np.histogram(x_ref, bins=edges)
        test_counts, _ = np.histogram(x_test, bins=edges)

        ref_pct = ref_counts.astype(float) / max(ref_counts.sum(), 1)
        test_pct = test_counts.astype(float) / max(test_counts.sum(), 1)

        # Replace zeros with smallest float to avoid log(0)
        ref_pct[ref_pct == 0] = sys.float_info.min
        test_pct[test_pct == 0] = sys.float_info.min

        psi_value = float(
            np.sum((test_pct - ref_pct) * np.log(test_pct / ref_pct))
        )
        return DataDriftResult(
            is_drift=psi_value > self._threshold,
            statistic=psi_value,
            distance=psi_value,
        )

    def get_params(self) -> Dict[str, Any]:
        return {
            "window_size": self._window_size,
            "num_bins": self._num_bins,
            "threshold": self._threshold,
            "correction": self._correction,
        }
