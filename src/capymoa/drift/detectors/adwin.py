from typing import Optional

from capymoa.drift.base_detector import MOADriftDetector

from moa.classifiers.core.driftdetection import (
    ADWINChangeDetector as _ADWINChangeDetector,
)


class ADWIN(MOADriftDetector):
    """ADWIN Drift Detector

    Example:
    --------

    >>> import numpy as np
    >>> from capymoa.drift.detectors import ADWIN
    >>> np.random.seed(0)
    >>> detector = ADWIN(delta=0.001)
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
    Change detected in data: 5 - at index: 1055

    Reference:
    ----------

    Bifet, Albert, and Ricard Gavalda. "Learning from time-changing data with adaptive windowing."
    Proceedings of the 2007 SIAM international conference on data mining.
    Society for Industrial and Applied Mathematics, 2007.

    """

    def __init__(self, delta: float = 0.002, CLI: Optional[str] = None):
        if CLI is None:
            CLI = f"-a {delta}"

        super().__init__(moa_detector=_ADWINChangeDetector(), CLI=CLI)

        self.delta = delta
        self.get_params()
