from typing import Optional

from capymoa.drift.base_detector import MOADriftDetector

from moa.classifiers.core.driftdetection import PageHinkleyDM as _PageHinkleyDM


class PageHinkley(MOADriftDetector):
    """Page-Hinkley Drift Detector

    Example:
    --------

    >>> import numpy as np
    >>> from capymoa.drift.detectors import PageHinkley
    >>> np.random.seed(0)
    >>>
    >>> detector = PageHinkley()
    >>>
    >>> data_stream = np.random.randint(2, size=2000)
    >>> for i in range(999, 2000):
    ...     data_stream[i] = np.random.randint(4, high=8)
    >>>
    >>> for i in range(2000):
    ...     detector.add_element(data_stream[i])
    ...     if detector.detected_change():
    ...         print('Change detected in data: ' + str(data_stream[i]) + ' - at index: ' + str(i))
    Change detected in data: 7 - at index: 1014
    Change detected in data: 7 - at index: 1685

    Reference:
    ----------

    Page. 1954. Continuous Inspection Schemes. Biometrika 41, 1/2 (1954),
    100-115.

    """

    def __init__(
        self,
        min_n_instances: int = 30,
        delta: float = 0.005,
        lambda_: float = 50.0,
        alpha: float = 0.9999,
        CLI: Optional[str] = None,
    ):
        if CLI is None:
            CLI = f"-n {min_n_instances} -d {delta} -l {lambda_} -a {alpha}"

        super().__init__(moa_detector=_PageHinkleyDM(), CLI=CLI)

        self.min_n_instances = min_n_instances
        self.delta = delta
        self.lambda_ = lambda_
        self.alpha = alpha
        self.get_params()
