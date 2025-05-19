from typing import List, Union, Tuple, Dict

import numpy as np
import pandas as pd

ArrayLike = Union[List[int], np.ndarray[int]]


class EvaluateDetector:
    """Evaluate Drift Detector

    Evaluate drift detection Methods based on known drift locations

    References:
    - Cerqueira, Vitor, Heitor Murilo Gomes, and Albert Bifet.
    "Unsupervised concept drift detection using a student–teacher approach."
    Discovery Science: 23rd International Conference, DS 2020, Thessaloniki, Greece, October 19–21, 2020
    - Bifet, A.: Classifier concept drift detection and the illusion of progress.
    In: International Conference on Artificial Intelligence and Soft Computing. pp. 715–725. Springer (2017)

    Example usages:

    >>> import numpy as np
    >>> from capymoa.drift.detectors import ADWIN
    >>> from capymoa.drift.eval_detector import EvaluateDetector
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
    >>>
    >>> eval = EvaluateDetector(max_delay=200)
    >>> print(eval.calc_performance(preds, trues))
    mean_time_to_detect           24.0
    missed_detection_ratio         0.0
    mean_time_btw_false_alarms     NaN
    no_alarms_per_episode          0.0
    dtype: float64

    """

    def __init__(self, max_delay: int):
        """

        :param max_delay: Maximum number of instances to wait for a detection
            [after which the drift becomes obvious and the detector is considered to have missed the change]
        """

        self.max_delay = max_delay

        self.results = pd.Series(
            {
                "mean_time_to_detect": -1,
                "missed_detection_ratio": -1,
                "mean_time_btw_false_alarms": -1,
                "alarms_per_ep_mean": -1,
            }
        )

        self.metrics = []

    def calc_performance(self, preds: ArrayLike, trues: ArrayLike) -> pd.Series:
        """

        :param preds: (array): detection location (index) values
        :param trues: (array): actual location (index) of drift. For drifts based on an interval (e.g gradual drifts),
            the current approach is to define it using the starting location and the max_delay parameter
        :return: pd.Series with a performance summary
        """

        self._check_arrays(preds, trues)

        eps = self._get_drift_episodes(preds, trues)

        for ep in eps:
            mtfa, n_alarms = self.calc_false_alarms(**ep)
            det_delay, detected_flag = self.calc_detection_delay(**ep)

            self.metrics.append(
                {
                    "mtfa": mtfa,
                    "n_alarms": n_alarms,
                    "delay": det_delay,
                    "detected": detected_flag,
                }
            )

        self.update_metrics()

        return self.results

    def update_metrics(self):
        df = pd.DataFrame(self.metrics)

        self.results = pd.Series(
            {
                "mean_time_to_detect": df["delay"].mean(),
                "missed_detection_ratio": 1 - df["detected"].mean(),
                "mean_time_btw_false_alarms": df["mtfa"].mean(),
                "no_alarms_per_episode": df["n_alarms"].mean(),
            }
        )

    def _get_drift_episodes(self, preds: ArrayLike, trues: ArrayLike) -> List[Dict]:
        if not isinstance(preds, np.ndarray):
            preds = np.asarray(preds)

        if not isinstance(trues, np.ndarray):
            trues = np.asarray(trues)

        last_cut = 0
        drift_episodes = []
        for true in trues:
            episode_preds = preds[preds <= (true + self.max_delay)]
            episode_preds = episode_preds[episode_preds > last_cut]
            episode_preds -= last_cut

            drift_episodes.append({"preds": episode_preds, "true": true - last_cut})

            last_cut = true + self.max_delay

        return drift_episodes

    def _check_arrays(self, preds: ArrayLike, trues: ArrayLike):
        assert len(trues) > 0, "No drift points given"

        if not isinstance(preds, np.ndarray):
            preds = np.asarray(preds)

        if not isinstance(trues, np.ndarray):
            trues = np.asarray(trues)

        if len(preds) > 1:
            tot_neg_alarms = np.sum(np.diff(preds) < 0)
            if tot_neg_alarms > 0:
                raise ValueError("Provide an ordered list of detections")

        if len(trues) > 1:
            tot_neg_drifts = np.sum(np.diff(trues) < 0)
            if tot_neg_drifts > 0:
                raise ValueError("Provide an ordered list of drift points")

    @staticmethod
    def calc_false_alarms(preds: ArrayLike, true: int) -> Tuple[float, int]:
        """

        :param preds: detection points
        :param true: actual drift point

        :return: tuple, (float: mean time between false alarms, int: no. false alarms)
        """

        if not isinstance(preds, np.ndarray):
            preds = np.asarray(preds)

        alarms_before_drift = preds[preds < true]
        n_alarms = len(alarms_before_drift)

        if n_alarms == 1:
            mean_time_btw_alarms = alarms_before_drift[0]
        elif n_alarms == 0:
            mean_time_btw_alarms = np.nan
        else:
            mean_time_btw_alarms = np.nanmean(np.diff(alarms_before_drift))

        return mean_time_btw_alarms, n_alarms

    @staticmethod
    def calc_detection_delay(preds: ArrayLike, true: int) -> Tuple[int, int]:
        """

        :param preds: detection points
        :param true: actual drift point

        :return: detection delay (number of instances), and a flag for whether drift is detected

        """

        if not isinstance(preds, np.ndarray):
            preds = np.asarray(preds)

        preds_after_drift = preds[preds > true]

        if len(preds_after_drift) > 0:
            first_alarm = preds_after_drift[0]
            delay_ = first_alarm - true
            detected_drift = 1
        else:
            # no detection
            delay_ = np.nan
            detected_drift = 0

        return delay_, detected_drift
