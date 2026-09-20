from moa.classifiers.core.driftdetection import EDDM as _EDDM

from capymoa.drift.base_detector import MOADriftDetector


class EDDM(MOADriftDetector):
    """Early Drift Detection Method (EDDM) Drift Detector

    EDDM monitors the distance between two consecutive errors. A drop in the average
    distance between errors indicates drift.

    EDDM complements DDM. DDM tracks the error rate. EDDM tracks how far apart
    errors are. The two often disagree: EDDM is more stable on noisy streams,
    DDM reacts faster when the error rate jumps.

    The detector has no tuning parameters. MOA uses the fixed warning and drift
    thresholds from the original paper.

    Example:
    --------

    >>> from capymoa.drift.detectors import EDDM
    >>> detector = EDDM()
    >>>
    >>> data_stream = [0] * 1000 + [1] * 1000
    >>>
    >>> for i, x in enumerate(data_stream):
    ...     detector.add_element(x)
    ...     if detector.detected_change():
    ...         print('Change detected in data: ' + str(x) + ' - at index: ' + str(i))
    Change detected in data: 1 - at index: 1030

    Reference:
    ----------

    Baena-García, M., del Campo-Ávila, J., Fidalgo, R., Bifet, A., Gavaldà, R., &
    Morales-Bueno, R. (2006). Early drift detection method. Fourth International
    Workshop on Knowledge Discovery from Data Streams, 6, 77-86.

    """

    _moa_detector_type = _EDDM

    def __init__(self):
        """Create an EDDM drift detector.

        EDDM has no hyper-parameters. MOA uses the thresholds from the original paper.
        """
        super().__init__()
