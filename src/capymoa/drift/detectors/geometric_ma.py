from typing import Optional

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

    def __init__(
        self,
        min_n_instances: int = 30,
        lambda_: float = 1.0,
        alpha: float = 0.99,
        CLI: Optional[str] = None,
    ):
        if CLI is None:
            CLI = f"-n {min_n_instances} -l {lambda_} -a {alpha}"

        super().__init__(moa_detector=_GeometricMovingAverageDM(), CLI=CLI)

        self.min_n_instances = min_n_instances
        self.lambda_ = lambda_
        self.alpha = alpha
        self.get_params()
