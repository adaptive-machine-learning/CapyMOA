from capymoa.drift.base_detector import MOADriftDetector

from moa.classifiers.core.driftdetection import (
    GeometricMovingAverageDM as _GeometricMovingAverageDM,
)


class GeometricMovingAverage(MOADriftDetector):
    """Geometric Moving Average Test Drift Detector

    Example:
    --------

    >>> import numpy as np
    >>> from capymoa.drift.detectors import GeometricMovingAverage
    >>> np.random.seed(0)
    >>>
    >>> detector = GeometricMovingAverage()
    >>>
    >>> data_stream = np.random.randint(2, size=2000)
    >>> for i in range(999, 2000):
    ...     data_stream[i] = np.random.randint(4, high=8)
    >>>
    >>> for i in range(2000):
    ...     detector.add_element(data_stream[i])
    ...     if detector.detected_change():
    ...         print('Change detected in data: ' + str(data_stream[i]) + ' - at index: ' + str(i))
    Change detected in data: 4 - at index: 1023

    """

    _moa_detector_type = _GeometricMovingAverageDM

    def __init__(
        self,
        min_n_instances: int = 30,
        lambda_: float = 1.0,
        alpha: float = 0.99,
    ):
        """Create a Geometric Moving Average drift detector.

        :param min_n_instances: Minimum number of instances to observe before change
            detection is enabled. Defaults to 30.
        :param lambda_: Detection threshold for the geometric moving-average statistic;
            higher values require stronger evidence before reporting change. Defaults to
            1.0.
        :param alpha: Smoothing factor for the geometric moving-average statistic.
            Values closer to 1.0 emphasise past observations and react more slowly to
            recent changes. Defaults to 0.99.
        """
        cli = f"-n {min_n_instances} -l {lambda_} -a {alpha}"
        super().__init__(cli=cli)
