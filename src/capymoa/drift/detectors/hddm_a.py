from typing import Literal, Optional

from capymoa.drift.base_detector import MOADriftDetector

from moa.classifiers.core.driftdetection import HDDM_A_Test as _HDDM_A_Test


class HDDMAverage(MOADriftDetector):
    """Average Hoeffding's bounds Drift Detector

    Example usages:
    ---------------

    >>> import numpy as np
    >>> from capymoa.drift.detectors import HDDMAverage
    >>> np.random.seed(0)
    >>>
    >>> detector = HDDMAverage(drift_confidence=1e-10)
    >>>
    >>> data_stream = np.random.randint(2, size=2000)
    >>> for i in range(999, 2000):
    ...     data_stream[i] = np.random.randint(4, high=8)
    >>>
    >>> for i in range(2000):
    ...     detector.add_element(data_stream[i])
    ...     if detector.detected_change():
    ...         print("Change detected in data: " + str(data_stream[i]) + " - at index: " + str(i))
    Change detected in data: 5 - at index: 999
    Change detected in data: 7 - at index: 1019
    Change detected in data: 7 - at index: 1330

    Reference:
    ----------

    Frias-Blanco, Isvani, et al. "Online and non-parametric drift
    detection methods based on Hoeffding's bounds." IEEE Transactions on
    Knowledge and Data Engineering 27.3 (2014): 810-823.

    """

    TEST_TYPES = ["Two-sided", "One-sided"]

    def __init__(
        self,
        drift_confidence: float = 0.001,
        warning_confidence: float = 0.005,
        test_type: Literal["Two-sided", "One-sided"] = "Two-sided",
        CLI: Optional[str] = None,
    ):
        """
        :param drift_confidence: Significance level for drift detection (p-value threshold).
        :param warning_confidence: Significance level for warning detection (p-value threshold).
        :param test_type: The type of statistical hypothesis test to use.

            * ``"Two-sided"`` detects drift in both directions, i.e. both increases and
              decreases in the monitored average. Use this when the direction of change
              is unknown.
            * ``"One-sided"`` only detects increases in the monitored average (i.e.
              performance degradation). Use this when only positive drift is relevant.

        :param CLI: (Advanced) Override the CLI arguments passed directly to the
            underlying MOA detector, bypassing all other parameters.
        """
        assert test_type in self.TEST_TYPES, "Wrong test type"

        if CLI is None:
            CLI = f"-d {drift_confidence} -w {warning_confidence} -t {test_type}"

        super().__init__(moa_detector=_HDDM_A_Test(), CLI=CLI)

        self.drift_confidence = drift_confidence
        self.warning_confidence = warning_confidence
        self.test_type = test_type
        self.get_params()
