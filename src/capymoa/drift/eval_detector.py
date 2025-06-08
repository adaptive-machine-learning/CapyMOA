import warnings
from dataclasses import dataclass
from typing import List, Union, Tuple, Dict, Optional, Any, Sequence

import numpy as np

_ArrayOrTupleOf = Union[Sequence[int], Sequence[Tuple[int, int]], np.ndarray]


@dataclass
class DriftDetectionMetrics:
    """Metrics for evaluating drift detection performance."""

    fp: int
    """False positives (incorrect detections)."""
    tp: int
    """True positives (correct detections)."""
    fn: int
    """False negatives (missed drifts)."""
    precision: float
    """Precision score ``(tp / (tp + fp))``."""
    recall: float
    """Recall score ``(tp / (tp + fn))``."""
    episode_recall: float
    """Recall score for drift episodes (in this case, a correct prediction of a
    drift episode is counted as 1 true positive)."""
    f1: float
    """F1 score (harmonic mean of precision and recall)."""
    mdt: float
    """Mean time to detect successful detections."""
    far: float
    """False alarm rate per :py:attr:`EvaluateDriftDetector.rate_period` instances"""
    ar: float
    """Alarm rate per :py:attr:`EvaluateDriftDetector.rate_period` instances"""
    n_episodes: int
    """Total number of drift episodes"""
    n_alarms: int
    """Total number of alarms raised"""


class EvaluateDriftDetector:
    """A class to evaluate the performance of concept drift detectors.

    This class provides functionality to assess drift detection algorithms by comparing
    their predictions against ground truth drift points. Each drift point is represented
    as a tuple (start_location, end_location) to handle both abrupt and gradual drifts.

    Max delay is the maximum number of instances to wait for a detection after a drift
    occurs. For gradual drifts, this window starts from the drift end_location.
    If a detector fails to signal within this window, it is considered to have
    missed the drift (false negative). The max delay can be thought of as the time after
    which a drift becomes apparent even without a detector.

    Key Features:
        - Handles both abrupt and gradual drifts:
            * Abrupt drifts: ``start_location = end_location``, e.g., (100, 100)
            * Gradual drifts: ``start_location < end_location``, e.g., (100, 150)
        - Considers maximum acceptable detection delay
        - Calculates comprehensive performance metrics (precision, recall, F1)

    Examples:
        >>> import numpy as np
        >>>
        >>> from capymoa.drift.eval_detector import EvaluateDriftDetector
        >>>
        >>> preds = np.array([500, 1200, 1250, 2100])
        >>> trues = np.array([1000, 2000])
        >>>
        >>> eval = EvaluateDriftDetector(max_delay=200)
        >>> metrics = eval.calc_performance(preds=preds, trues=trues, tot_n_instances=200)
        >>> print(metrics.f1)
        0.6666666666666666

        >>> # Example with actual detector
        >>> import numpy as np
        >>> from capymoa.drift.detectors import ADWIN
        >>> from capymoa.drift.eval_detector import EvaluateDriftDetector
        >>>
        >>> detector = ADWIN(delta=0.001)
        >>>
        >>> data_stream = np.random.randint(2, size=2000)
        >>> for i in range(999, 2000):
        ...     data_stream[i] = np.random.randint(4, high=8)
        >>>
        >>> for i in range(2000):
        ...     detector.add_element(data_stream[i])
        >>>
        >>> trues = np.array([1000])
        >>> preds = detector.detection_index
        >>> evaluator = EvaluateDriftDetector(max_delay=50)
        >>> metrics = evaluator.calc_performance(trues=trues, preds=preds, tot_n_instances=2000)
        >>> print(metrics.f1)
        1.0

    """

    def __init__(
        self, max_delay: int, rate_period: int = 1000, max_early_detection: int = 0
    ):
        """
        Initialize the drift detector evaluator.

        :param int max_delay: Maximum number of instances to wait for a detection after a drift
            occurs. For gradual drifts, this window starts from the drift end_location.
            If a detector fails to signal within this window, it is considered to have
            missed the drift (false negative).
        :param int max_early_detection: Maximum number of instances before a drift starts where
            a detection is still considered valid. For detections that occur earlier than
            this window, they are considered false positives.
        :param int rate_period: The number of instances in a relevant time period. This number is
            used to calculate the false alarm rate and alarm rate.
            Default is 1000.

        :raises ValueError: If max_delay is not a positive integer.

        .. note::
            - The ``max_delay`` parameter is important for evaluating both the accuracy and timing
              of drift detection.
            - For gradual drifts (where ``start_location != end_location``), the detection
              window extends from (``start_location - max_delay``) to (``end_location + max_delay``).
            - For abrupt drifts (where ``start_location == end_location``), the detection
              window is (``drift_point - max_delay``) to (``drift_point + max_delay``).
        """
        self._validate_parameters(max_delay, rate_period)
        #: Maximum allowable delay for drift detection.
        self.max_delay: int = max_delay
        #: Maximum allowable earliness for drift detection.
        self.max_early_detection: int = max_early_detection
        #: The period used for calculating rates (e.g., per 1000 instances).
        self.rate_period: int = rate_period
        #: Latest calculated performance metrics.
        self.metrics: Optional[DriftDetectionMetrics] = None

    def calc_performance(
        self,
        trues: _ArrayOrTupleOf,
        preds: _ArrayOrTupleOf,
        tot_n_instances: int,
        drift_episodes: Optional[List[Dict[str, Any]]] = None,
    ) -> DriftDetectionMetrics:
        """
        Calculate performance metrics for drift detection.

        Evaluates drift detection performance by comparing predicted drift points against
        true drift points, considering a maximum allowable delay. Calculates various metrics
        including precision, recall, F1-score, mean time to detect (MTD), and false alarm rate.

        This method supports two modes of operation:

        1. Pass ``trues``, ``preds``, and ``tot_n_instances`` to evaluate from raw data.
        2. Pass ``drift_episodes`` and ``tot_n_instances`` to evaluate from pre-computed episodes.

        :param trues: Array-like of true drift points represented as (start, end) tuples
            indicating drift intervals. Required if ``drift_episodes`` is not provided.
        :param preds: Array-like of predicted drift points (indices) where the detector
            signaled a drift. Required if ``drift_episodes`` is not provided.
        :param tot_n_instances: Total number of instances in the data stream, used to
            calculate alarm rate and false alarm rate. Always required.
        :param drift_episodes: Optional list of drift episodes. If provided, the ``trues`` and ``preds``
            parameters will be ignored with a warning.

        :returns: :class:`DriftDetectionMetrics` object containing the calculated metrics

        :raises ValueError: If arrays are not ordered or contain invalid values,
            if ``tot_n_instances`` is not positive, or if neither ``drift_episodes`` nor
            (``trues`` and ``preds``) are provided.
        :raises AssertionError: If no drift points are given.
        """
        if tot_n_instances <= 0:
            raise ValueError("Total number of instances must be positive")

        if drift_episodes is not None:
            if not drift_episodes:
                raise ValueError("drift_episodes cannot be empty")

            if trues is not None or preds is not None:
                warnings.warn(
                    "Both drift_episodes and trues/preds were provided. "
                    "The trues and preds parameters will be ignored.",
                    UserWarning,
                )
            drift_eps = drift_episodes
        else:
            if trues is None or preds is None:
                raise ValueError(
                    "Either drift_episodes or both trues and preds must be provided"
                )

            self._check_arrays(trues, preds)
            drift_eps = self._get_drift_episodes(trues=trues, preds=preds)

        fp, tp, fn = 0, 0, 0
        etp = 0  # episode true positives
        detection_times: List[float] = []
        n_episodes, n_alarms = 0, 0

        for episode in drift_eps:
            n_episodes += 1
            drift_detected = False
            episode_detection_time = np.nan

            try:
                drift_start, drift_end = episode["true"]
            except (ValueError, KeyError, TypeError) as e:
                raise ValueError(
                    f"Invalid episode format: {e}. Expected 'true' field with (start, end) tuple."
                )

            episode_preds = episode.get("preds", np.array([]))
            if not isinstance(episode_preds, np.ndarray):
                episode_preds = np.asarray(episode_preds)

            for pred in episode_preds:
                n_alarms += 1

                if (
                    drift_start - self.max_early_detection
                    <= pred
                    <= drift_end + self.max_delay
                ):
                    tp += 1
                    if not drift_detected:  # only counting first detection
                        drift_detected = True
                        episode_detection_time = pred - drift_start
                else:
                    fp += 1

            if drift_detected:
                etp += 1
                detection_times.append(episode_detection_time)
            else:
                fn += 1

        precision, recall, f1 = self._calc_classification_metrics(tp=tp, fp=fp, fn=fn)
        false_alarm_rate = (fp / max(1, tot_n_instances)) * self.rate_period
        alarm_rate = (n_alarms / max(1, tot_n_instances)) * self.rate_period
        mean_detection_time = np.nanmean(detection_times) if detection_times else np.nan
        ep_recall = etp / max(1, n_episodes)

        self.metrics = DriftDetectionMetrics(
            fp=fp,
            tp=tp,
            fn=fn,
            precision=precision,
            recall=recall,
            episode_recall=ep_recall,
            f1=f1,
            mdt=mean_detection_time,
            far=false_alarm_rate,
            ar=alarm_rate,
            n_episodes=n_episodes,
            n_alarms=n_alarms,
        )

        return self.metrics

    def _get_drift_episodes(
        self, trues: _ArrayOrTupleOf, preds: _ArrayOrTupleOf
    ) -> List[Dict[str, Any]]:
        """
        Process raw drift points and predictions into drift episodes.

        :param trues: Array-like of true drift points represented as (start, end) tuples.
        :param preds: Array-like of predicted drift points.
        :returns: List of dictionaries representing drift episodes, each containing:
            - 'preds': array of prediction times for this episode
            - 'true': tuple of (drift_start, drift_end) relative to episode start
        """
        preds_array = np.asarray(preds)
        trues_array = np.asarray(trues)

        if preds_array.ndim > 1:
            raise ValueError("preds must be a 1-dimensional array")

        if trues_array.ndim not in (1, 2) or (
            trues_array.ndim == 2 and trues_array.shape[1] != 2
        ):
            raise ValueError("trues must be an array of points or (start, end) tuples")

        if trues_array.ndim == 1:
            trues_array = np.column_stack((trues_array, trues_array))

        next_starting_point = 0
        drift_episodes = []

        for true in trues_array:
            drift_start, drift_end = true

            episode_preds = preds_array[preds_array > next_starting_point]
            episode_preds = episode_preds[episode_preds <= drift_end + self.max_delay]

            episode_preds = episode_preds - next_starting_point

            drift_episodes.append(
                {
                    "preds": episode_preds,
                    "true": (
                        drift_start - next_starting_point,
                        drift_end - next_starting_point,
                    ),
                }
            )

            next_starting_point = drift_end + self.max_delay

        return drift_episodes

    @staticmethod
    def _validate_parameters(max_delay: int, rate_period: int) -> None:
        """
        Validate constructor parameters.

        :param int max_delay: Maximum detection delay to allow.
        :param int rate_period: Period for rate calculations.
        """
        if not isinstance(max_delay, int) or max_delay <= 0:
            raise ValueError("max_delay must be a positive integer")

        if not isinstance(rate_period, int) or rate_period <= 0:
            raise ValueError("rate_period must be a positive integer")

    @staticmethod
    def _check_arrays(trues: _ArrayOrTupleOf, preds: _ArrayOrTupleOf) -> None:
        """
        Validate input arrays for consistency and correctness.

        :param trues: Array of drift points (start, end) tuples.
        :param preds: Array of detection points.
        """
        if trues is None:
            raise ValueError("No drift points given")

        preds_array = np.asarray(preds)
        trues_array = np.asarray(trues)

        if preds_array.ndim > 1:
            raise ValueError("preds must be a 1-dimensional array of detection points")

        if trues_array.ndim not in (1, 2):
            raise ValueError(
                "trues must be a 1D array of points or a 2D array of (start, end) tuples"
            )

        if trues_array.ndim == 2 and trues_array.shape[1] != 2:
            raise ValueError(
                "When trues is 2D, it must have exactly 2 columns for (start, end)"
            )

        if len(preds_array) > 1:
            diffs = np.diff(preds_array)
            if np.any(diffs < 0):
                raise ValueError("Provide an ordered list of detections")

        if len(trues_array) > 1:
            if trues_array.ndim == 1:
                tot_neg_drifts = np.sum(np.diff(trues_array) < 0)
                if tot_neg_drifts > 0:
                    raise ValueError("Provide an ordered list of drift points")
            else:
                tot_neg_drifts = np.sum(np.diff(trues_array[:, 0]) < 0)
                if tot_neg_drifts > 0:
                    raise ValueError("Provide an ordered list of drift intervals")

                if np.any(trues_array[:, 0] > trues_array[:, 1]):
                    raise ValueError("For each drift interval, start must be <= end")

    @staticmethod
    def _calc_classification_metrics(
        tp: int, fp: int, fn: int
    ) -> Tuple[float, float, float]:
        """
        Calculate precision, recall, and F1 score with safeguards against division by zero.

        :param int tp: True positive count
        :param int fp: False positive count
        :param int fn: False negative count

        :returns: Tuple of (precision, recall, f1_score)
        """
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0

        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

        if precision + recall == 0:
            f1_score = 0.0
        else:
            f1_score = 2.0 * (precision * recall) / (precision + recall)

        return precision, recall, f1_score
