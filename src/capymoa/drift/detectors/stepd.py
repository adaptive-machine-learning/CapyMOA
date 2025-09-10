from typing import Optional

from capymoa.drift.base_detector import MOADriftDetector

from moa.classifiers.core.driftdetection import STEPD as _STEPD


class STEPD(MOADriftDetector):
    """Statistical Test of Equal Proportions Drift Detector

    Example:
    --------

    >>> import numpy as np
    >>> from capymoa.drift.detectors import STEPD
    >>> np.random.seed(0)
    >>>
    >>> detector = STEPD()
    >>>
    >>> data_stream = np.random.randint(2, size=2000)
    >>> for i in range(999, 2000):
    ...     data_stream[i] = np.random.randint(4, high=8)
    >>>
    >>> for i in range(2000):
    ...     detector.add_element(data_stream[i])
    ...     if detector.detected_change():
    ...         print('Change detected in data: ' + str(data_stream[i]) + ' - at index: ' + str(i))
    Change detected in data: 6 - at index: 1001

    Reference:
    ----------

    Nishida, Kyosuke, and Koichiro Yamauchi. "Detecting concept drift using
    statistical testing." International conference on discovery science. Berlin,
    Heidelberg: Springer Berlin Heidelberg, 2007.
    """

    def __init__(
        self,
        window_size: int = 30,
        alpha_drift: float = 0.003,
        alpha_warning: float = 0.05,
        CLI: Optional[str] = None,
    ):
        if CLI is None:
            CLI = f"-r {window_size} -o {alpha_drift} -w {alpha_warning}"

        super().__init__(moa_detector=_STEPD(), CLI=CLI)

        self.window_size = window_size
        self.alpha_drift = alpha_drift
        self.alpha_warning = alpha_warning
        self.get_params()
