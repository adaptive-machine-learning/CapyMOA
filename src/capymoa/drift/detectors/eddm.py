from capymoa.drift.base_detector import MOADriftDetector

from moa.classifiers.core.driftdetection import EDDM as _EDDM


class EDDM(MOADriftDetector):
    """Early Drift Detection Method Drift Detector

    Reference:
    Baena-Garcıa, M., del Campo-Ávila, J., Fidalgo, R., Bifet, A., Gavalda, R., & Morales-Bueno, R. (2006, September).
    Early drift detection method.
    In Fourth international workshop on knowledge discovery from data streams (Vol. 6, pp. 77-86).

    Example usages:

    >>> import numpy as np
    >>> from capymoa.drift.detectors import EDDM
    >>>
    >>> detector = EDDM()
    >>>
    >>> data_stream = np.random.randint(2, size=2000)
    >>> for i in range(999, 2000):
    >>>     data_stream[i] = np.random.randint(4, high=8)
    >>>
    >>> for i in range(2000):
    >>>     detector.add_element(data_stream[i])
    >>>     if detector.detected_change():
    >>>         print('Change detected in data: ' + str(data_stream[i]) + ' - at index: ' + str(i))

    """

    def __init__(self):
        super().__init__(
            moa_detector=_EDDM(),
            CLI=None
        )

        self.get_params()
