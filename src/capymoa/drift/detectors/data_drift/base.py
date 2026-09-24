"""Base class for data drift detectors.

Data drift detectors compare a sliding window of recent observations against
a fixed reference distribution. Provide that reference in one of two ways:

* Call :meth:`fit` with a reference sample, then stream with
  :meth:`add_element`.
* Set ``auto_fit_samples`` at construction. That many initial
  :meth:`add_element` calls build the reference; no explicit :meth:`fit`
  is needed.

The reference stays fixed unless you call :meth:`fit` again along the
stream. Once the test window is full the detector runs a statistical
comparison and exposes the result through :meth:`detected_change`,
:attr:`result`, and the inherited :attr:`detection_index` list.

This differs from concept drift detectors, which track a scalar error signal
and need no reference data (``REQUIRES_FIT = False``).
"""

import sys
from abc import abstractmethod
from collections import deque
from collections.abc import Hashable, Sequence
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np

from capymoa.drift.base_detector import BaseDriftDetector


def _bin_probabilities(
    x_ref: np.ndarray, x_test: np.ndarray, num_bins: int
) -> tuple[np.ndarray, np.ndarray]:
    """Compute bin probabilities for reference and test samples.

    Bin edges span the combined range of both samples so that every
    observation falls inside a bin. Zeros are replaced with
    ``sys.float_info.min`` to avoid ``log(0)`` and division by zero.

    :param x_ref: 1-D reference sample.
    :param x_test: 1-D test sample.
    :param num_bins: Number of equal-width bins.
    :returns: ``(ref_probs, test_probs)`` each of shape ``(num_bins,)``.
    """
    combined = np.concatenate([x_ref, x_test])
    edges = np.linspace(combined.min(), combined.max(), num_bins + 1)
    ref_counts, _ = np.histogram(x_ref, bins=edges)
    test_counts, _ = np.histogram(x_test, bins=edges)
    ref_counts = ref_counts.astype(float)
    test_counts = test_counts.astype(float)
    ref_counts[ref_counts == 0] = sys.float_info.min
    test_counts[test_counts == 0] = sys.float_info.min
    ref_probs = ref_counts / ref_counts.sum()
    test_probs = test_counts / test_counts.sum()
    return ref_probs, test_probs


@dataclass
class DataDriftResult:
    """Result returned by a data-drift comparison.

    Every comparison produces a ``statistic``. Statistical-test detectors
    also set ``p_value``; distance-based detectors leave it as ``None``
    and may set ``distance`` instead.

    For univariate tests on multivariate data, ``feature_statistics``,
    ``feature_p_values``, and ``feature_is_drift`` map feature index
    to that feature's value. The top-level ``statistic`` and
    ``p_value`` are aggregates (max statistic, min p-value).
    ``is_drift`` is the overall decision: for tests that produce
    p-values, after multiple-testing correction; for distance-based
    tests (no p-value), the per-feature decisions are combined without
    correction (see :meth:`BaseDataDriftDetector._test`).
    """

    is_drift: bool
    """Overall drift decision (any feature). After correction for
    p-value tests; uncorrected for distance-based tests without
    p-values."""
    statistic: float
    """Aggregate test statistic (max across features for univariate)."""
    p_value: float | None = None
    """Aggregate p-value (min across features for univariate)."""
    distance: float | None = None
    """Distance metric, if applicable (e.g. MMD, EMD)."""
    feature_statistics: dict[Hashable, float] | None = None
    """Per-feature statistics. Keys are feature names when available,
    otherwise integer indices. ``None`` for multivariate tests."""
    feature_p_values: dict[Hashable, float] | None = None
    """Per-feature p-values. Keys are feature names when available,
    otherwise integer indices. ``None`` when the test has no p-value
    or for multivariate tests."""
    feature_is_drift: dict[Hashable, bool] | None = None
    """Per-feature drift flags. Keys are feature names when available,
    otherwise integer indices. ``None`` for multivariate tests."""


class BaseDataDriftDetector(BaseDriftDetector):
    """Base class for detectors that monitor the input data distribution.

    These detectors have :attr:`REQUIRES_FIT` set to ``True``: they need
    a reference distribution. Provide it with :meth:`fit`, or construct
    with ``auto_fit_samples`` so :meth:`add_element` collects it.

    Subclasses set the class variable :attr:`IS_UNIVARIATE` and implement:

    * :meth:`_fit` -- store or preprocess the reference data.
    * :meth:`_test` -- run the test (on one feature when univariate, on
      all features when multivariate).
    * :meth:`get_params` -- return detector hyper-parameters.

    The base class handles the sliding test window, feature-wise looping for
    univariate tests, and detection bookkeeping. For univariate tests that
    produce p-values, it also applies Bonferroni correction
    [#rabanser2019]_ across features. Univariate tests that compare a
    distance or score to a fixed threshold instead (no p-value) supply
    their own per-feature decisions directly; ``correction`` has no effect
    on those.

    By default the reference window is fixed after :meth:`fit` (or after
    auto-fit) [#cerqueira2023]_ [#lukats2025]_. Call :meth:`fit` again
    along the stream to use a sliding reference.

    .. [#rabanser2019] Rabanser, S., Günnemann, S., and Lipton, Z. (2019).
        Failing loudly: An empirical study of methods for detecting dataset
        shift. Advances in Neural Information Processing Systems, 32.
    .. [#cerqueira2023] Cerqueira, V., Gomes, H. M., Bifet, A., and Torgo,
        L. (2023). STUDD: A student-teacher method for unsupervised concept
        drift detection. Machine Learning, 112(11), 4351-4378.
    .. [#lukats2025] Lukats, D., Zielinski, O., Hahn, A., and Stahl, F.
        (2025). A benchmark and survey of fully unsupervised concept drift
        detectors on real-world data streams. International Journal of Data
        Science and Analytics, 19(1), 1-31.
    """

    REQUIRES_FIT: bool = True
    """Data drift detectors need a reference distribution. Call
    :meth:`fit` or set ``auto_fit_samples`` so :meth:`add_element`
    can collect it."""

    IS_UNIVARIATE: bool = True
    """If ``True`` the test is applied to each feature separately and
    results are combined. If ``False`` the test runs on the joint
    distribution."""

    def __init__(
        self,
        window_size: int,
        alpha: float = 0.05,
        correction: Literal["bonferroni", "none"] = "bonferroni",
        auto_fit_samples: int | None = None,
    ):
        """Create a data drift detector.

        :param window_size: Number of observations to collect before
            running a comparison against the reference. The detector
            returns no result (and ``detected_change`` is ``False``)
            until the window is full.
        :param alpha: Significance level. For p-value tests, drift is
            declared when the (corrected) p-value falls below ``alpha``.
        :param correction: Multiple-testing correction for univariate
            tests that produce p-values, across features. ``"bonferroni"``
            (default) divides ``alpha`` by the number of features;
            ``"none"`` uses ``alpha`` directly. Ignored for multivariate
            tests, and for univariate tests that compare a distance or
            score to a fixed ``threshold`` instead of a p-value (those
            supply their own per-feature decisions, uncorrected).
        :param auto_fit_samples: If set, the first *auto_fit_samples*
            observations are used as the reference (auto-fit mode).
            No explicit :meth:`fit` call is needed; :meth:`add_element`
            collects the reference. If ``None`` (default), :meth:`fit`
            must be called before :meth:`add_element` or :meth:`compare`.
        :raises ValueError: If *window_size* is not a positive integer,
            *alpha* is not in ``(0, 1]``, *correction* is unknown, or
            *auto_fit_samples* is not a positive integer when set.
        """
        if not isinstance(window_size, int) or window_size <= 0:
            raise ValueError("window_size must be a positive integer")
        if not 0.0 < alpha <= 1.0:
            raise ValueError("alpha must be in (0, 1]")
        if correction not in ("bonferroni", "none"):
            raise ValueError("correction must be 'bonferroni' or 'none'")
        if auto_fit_samples is not None and (
            not isinstance(auto_fit_samples, int) or auto_fit_samples <= 0
        ):
            raise ValueError("auto_fit_samples must be a positive integer")

        super().__init__()
        self.in_warning_zone = False
        self._X_ref: np.ndarray | None = None
        self._n_features: int | None = None
        self._feature_names: list[str] | None = None
        self._window_size: int = window_size
        self._window: deque = deque(maxlen=window_size)
        self._result: DataDriftResult | None = None
        self._alpha: float = alpha
        self._correction: str = correction
        self._auto_fit_samples: int | None = auto_fit_samples
        self._ref_buffer: list[np.ndarray] | None = (
            [] if auto_fit_samples is not None else None
        )

    @property
    def X_ref(self) -> np.ndarray | None:
        """The reference data set with ``fit``."""
        return self._X_ref

    @property
    def n_features(self) -> int | None:
        """Number of features in the reference data, or ``None`` before fit."""
        return self._n_features

    @property
    def feature_names(self) -> list[str] | None:
        """Feature names passed to :meth:`fit`, or ``None``."""
        return self._feature_names

    @property
    def window_size(self) -> int:
        """Size of the sliding test window."""
        return self._window_size

    @property
    def alpha(self) -> float:
        """Significance level for drift decisions."""
        return self._alpha

    @property
    def correction(self) -> str:
        """Multiple-testing correction across features."""
        return self._correction

    @property
    def auto_fit_samples(self) -> int | None:
        """Number of samples for auto-fit, or ``None`` if explicit fit."""
        return self._auto_fit_samples

    @property
    def is_fitted(self) -> bool:
        """Whether the reference distribution has been set.

        ``True`` after :meth:`fit`, or after :meth:`add_element` has
        collected :attr:`auto_fit_samples` observations.
        """
        return self._X_ref is not None

    @property
    def result(self) -> DataDriftResult | None:
        """Most recent comparison result, or ``None`` during warm-up."""
        return self._result

    def fit(
        self,
        X: np.ndarray | Any,
        feature_names: Sequence[str] | None = None,
    ) -> None:
        """Set the reference distribution.

        This detector has :attr:`REQUIRES_FIT` set to ``True``. Call
        :meth:`fit` before :meth:`add_element` or :meth:`compare`, unless
        ``auto_fit_samples`` was set so :meth:`add_element` can collect
        the reference. Calling :meth:`fit` again replaces the reference
        (a sliding reference window).

        :param X: Reference data, shape ``(n_samples,)`` or
            ``(n_samples, n_features)``. May also be a pandas DataFrame,
            in which case column names are extracted automatically.
        :param feature_names: Optional names for each feature. If
            provided, must have length equal to the number of features.
            Overrides column names extracted from a DataFrame.
            When set, per-feature dicts in :class:`DataDriftResult` use
            these names as keys instead of integer indices.
        :raises ValueError: If *X* is empty or *feature_names* length
            does not match the number of features.
        """

        if feature_names is None and hasattr(X, "columns"):
            feature_names = list(X.columns)
        X = np.asarray(X)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        if X.shape[0] == 0:
            raise ValueError("Reference data must not be empty")
        n_features = X.shape[1]
        if feature_names is not None:
            if len(feature_names) != n_features:
                raise ValueError(
                    f"feature_names has {len(feature_names)} entries but "
                    f"X has {n_features} features"
                )
            self._feature_names = list(feature_names)
        else:
            self._feature_names = None
        self._n_features = n_features
        self._ref_buffer = None  # no longer collecting
        self._fit(X)

    def add_element(self, element: float | np.ndarray) -> None:
        """Add one observation and check for drift.

        The observation is appended to a sliding window of size
        :attr:`window_size`. Once full, the detector compares the window
        against the reference on every call.

        If the detector is not yet fitted and :attr:`auto_fit_samples`
        is set, those initial observations build the reference.
        No comparison happens until then. If it is not fitted and
        auto-fit is not enabled, this raises :class:`RuntimeError`.

        :param element: A single observation -- a scalar for univariate
            data, or a 1-D array for multivariate data.
        :raises RuntimeError: If the detector is not fitted and
            auto-fit mode is not enabled. Call :meth:`fit` first, or
            construct with ``auto_fit_samples``.
        :raises ValueError: If the observation has a different number of
            features than the reference.
        """
        element = np.asarray(element).ravel()

        if self._ref_buffer is not None:
            self._ref_buffer.append(element)
            self.idx += 1
            if len(self._ref_buffer) >= self._auto_fit_samples:  # type: ignore[operator]
                self.fit(np.vstack(self._ref_buffer))
            return

        if self._X_ref is None:
            raise RuntimeError(
                "This detector requires reference data. Call fit(X) first, "
                "or construct with auto_fit_samples so add_element() can "
                "collect the reference."
            )

        if element.size != self._n_features:
            raise ValueError(
                f"Expected {self._n_features} feature(s), got {element.size}"
            )
        self._window.append(element)
        self.idx += 1

        if len(self._window) >= self._window_size:
            X_test = np.vstack(self._window)
            self._result = self._run_comparison(self._X_ref, X_test)
            self.in_concept_change = self._result.is_drift
        else:
            self.in_concept_change = False
            self._result = None

        if self.in_concept_change:
            self.detection_index.append(self.idx)

    def compare(self, X_test: np.ndarray) -> DataDriftResult:
        """One-shot batch comparison against the reference.

        Unlike :meth:`add_element`, this does not update the internal
        window or detection history. It is useful for offline evaluation.
        ``auto_fit_samples`` does not apply here; call :meth:`fit` first.

        :param X_test: Test data with the same number of features as the
            reference.
        :raises RuntimeError: If :meth:`fit` has not been called.
        :raises ValueError: If *X_test* has a different number of features
            than the reference.
        :returns: Comparison result.
        """
        if self._X_ref is None:
            raise RuntimeError(
                "This detector requires reference data. Call fit(X) before "
                "compare(). auto_fit_samples only applies to add_element()."
            )
        X_test = np.asarray(X_test)
        if X_test.ndim == 1:
            X_test = X_test.reshape(-1, 1)
        if X_test.shape[1] != self._n_features:
            raise ValueError(
                f"Expected {self._n_features} feature(s), got {X_test.shape[1]}"
            )
        return self._run_comparison(self._X_ref, X_test)

    def reset(self, clean_history: bool = False) -> None:
        """Reset the detector state.

        :param clean_history: If ``True``, also clear the reference data
            and detection history. If ``False`` (default), only the
            sliding window and current result are cleared; the reference
            and detection indices are preserved.
        """
        super().reset(clean_history)
        self._window.clear()
        self._result = None
        if clean_history:
            self._X_ref = None
            self._n_features = None
            self._feature_names = None
            # Restore auto-fit buffer if auto-fit mode was configured
            if self._auto_fit_samples is not None:
                self._ref_buffer = []

    def _run_comparison(self, X_ref: np.ndarray, X_test: np.ndarray) -> DataDriftResult:
        """Route to univariate (per-feature) or multivariate comparison."""
        if self.IS_UNIVARIATE:
            return self._run_univariate(X_ref, X_test)
        return self._test(X_ref, X_test)

    def _run_univariate(self, X_ref: np.ndarray, X_test: np.ndarray) -> DataDriftResult:
        """Loop over features, apply the test, then combine results."""
        n_features = X_ref.shape[1]
        names = self._feature_names
        feature_statistics: dict[Hashable, float] = {}
        feature_p_values: dict[Hashable, float] = {}
        feature_is_drift_raw: dict[Hashable, bool] = {}
        has_p_values = True

        for i in range(n_features):
            key: Hashable = names[i] if names is not None else i
            r = self._test(X_ref[:, i], X_test[:, i])
            feature_statistics[key] = r.statistic
            feature_is_drift_raw[key] = r.is_drift
            if r.p_value is not None:
                feature_p_values[key] = r.p_value
            else:
                has_p_values = False

        if has_p_values:
            # Statistical tests: apply correction and threshold
            threshold = (
                self._alpha / n_features
                if self._correction == "bonferroni"
                else self._alpha
            )
            feature_p_values_out: dict[Hashable, float] | None = feature_p_values
            feature_is_drift = {k: p < threshold for k, p in feature_p_values.items()}
        else:
            # Distance-based tests: use is_drift from subclass
            feature_p_values_out = None
            feature_is_drift = feature_is_drift_raw

        return DataDriftResult(
            is_drift=any(feature_is_drift.values()),
            statistic=max(feature_statistics.values()),
            p_value=(
                min(feature_p_values_out.values())
                if feature_p_values_out is not None
                else None
            ),
            feature_statistics=feature_statistics,
            feature_p_values=feature_p_values_out,
            feature_is_drift=feature_is_drift,
        )

    @abstractmethod
    def _fit(self, X: np.ndarray) -> None:
        """Store or preprocess the reference data.

        Called by :meth:`fit` after validation. *X* is guaranteed to be a
        2-D ``np.ndarray`` with shape ``(n_samples, n_features)``.

        At minimum, implementations should set ``self._X_ref = X``.

        :meta public: include 'protected' method in documentation
        """

    @abstractmethod
    def _test(self, X_ref: np.ndarray, X_test: np.ndarray) -> DataDriftResult:
        """Run the underlying statistical test or distance measure.

        When ``IS_UNIVARIATE = True`` this receives **one feature** at a
        time: both arrays are 1-D with shapes ``(n_ref,)`` and
        ``(n_test,)``. The subclass should set ``statistic`` and
        ``p_value`` (if available) on the returned result.

        If ``p_value`` is set, the base class applies the configured
        ``correction`` and ignores the per-feature ``is_drift`` flag.
        Example::

            def _test(self, x_ref, x_test):
                stat, p = scipy.stats.ks_2samp(x_ref, x_test)
                return DataDriftResult(
                    is_drift=False, statistic=stat, p_value=p
                )

        If ``p_value`` is left ``None`` (distance- or score-based tests),
        the base class uses the per-feature ``is_drift`` flag directly and
        ``correction`` has no effect -- the subclass is responsible for its
        own per-feature decision (e.g. comparing a distance to a
        threshold).

        When ``IS_UNIVARIATE = False`` this receives **all features**:
        both arrays are 2-D with shapes ``(n_ref, n_features)`` and
        ``(n_test, n_features)``. The subclass is responsible for
        the full ``DataDriftResult`` including ``is_drift``.

        :meta public: include 'protected' method in documentation
        """

    @abstractmethod
    def get_params(self) -> dict[str, Any]:
        """Return the hyper-parameters of this detector."""
