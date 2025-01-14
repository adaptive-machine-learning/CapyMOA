from typing import Optional

from capymoa.drift.base_detector import MOADriftDetector

from moa.classifiers.core.driftdetection import EWMAChartDM as _EWMAChartDM


class EWMAChart(MOADriftDetector):
    """EWMA Charts Drift Detector

    Example:
    --------

    >>> import numpy as np
    >>> from capymoa.drift.detectors import EWMAChart
    >>> np.random.seed(0)
    >>>
    >>> detector = EWMAChart()
    >>>
    >>> data_stream = np.random.randint(2, size=2000)
    >>> for i in range(999, 2000):
    ...     data_stream[i] = np.random.randint(4, high=8)
    >>>
    >>> for i in range(2000):
    ...     detector.add_element(data_stream[i])
    ...     if detector.detected_change():
    ...         print('Change detected in data: ' + str(data_stream[i]) + ' - at index: ' + str(i))
    Change detected in data: 5 - at index: 999

    Reference:
    ----------

    Ross, Gordon J., et al. "Exponentially weighted moving average charts for
    detecting concept drift." Pattern recognition letters 33.2 (2012): 191-198.

    """

    def __init__(
        self, min_n_instances: int = 30, lambda_: float = 0.2, CLI: Optional[str] = None
    ):
        if CLI is None:
            CLI = f"-n {min_n_instances} -l {lambda_} "

        super().__init__(moa_detector=_EWMAChartDM(), CLI=CLI)

        self.min_n_instances = min_n_instances
        self.lambda_ = lambda_
        self.get_params()
