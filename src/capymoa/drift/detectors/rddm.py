from capymoa.drift.base_detector import MOADriftDetector

from moa.classifiers.core.driftdetection import RDDM as _RDDM


class RDDM(MOADriftDetector):
    """Reactive Drift Detection Method Drift Detector

    Example:
    --------

    >>> import numpy as np
    >>> from capymoa.drift.detectors import RDDM
    >>> np.random.seed(0)
    >>>
    >>> detector = RDDM()
    >>>
    >>> data_stream = np.random.randint(2, size=2000)
    >>> for i in range(999, 2000):
    ...     data_stream[i] = np.random.randint(4, high=8)
    >>>
    >>> for i in range(2000):
    ...     detector.add_element(data_stream[i])
    ...     if detector.detected_change():
    ...         print('Change detected in data: ' + str(data_stream[i]) + ' - at index: ' + str(i))
    Change detected in data: 6 - at index: 1003

    Reference:
    ----------

    Barros, R. S., Cabral, D. R., Gonçalves Jr, P. M., & Santos, S. G. (2017).
    RDDM: Reactive drift detection method. Expert Systems with Applications, 90, 344-355.

    """

    _moa_detector_type = _RDDM

    def __init__(
        self,
        min_n_instances: int = 129,
        warning_level: float = 1.773,
        drift_level: float = 2.258,
        max_size_concept: int = 40000,
        min_size_concept: int = 7000,
        warning_limit: int = 1400,
    ):
        """Create an RDDM drift detector.

        :param min_n_instances: Minimum number of instances before the detector starts
            monitoring for drift. Defaults to 129.
        :param warning_level: Multiplier applied to the minimum error estimate for
            entering the warning zone. Defaults to 1.773.
        :param drift_level: Multiplier applied to the minimum error estimate for
            declaring drift. Defaults to 2.258.
        :param max_size_concept: Maximum number of instances allowed in a stable concept
            before forcing an RDDM reset when no warning is active. Defaults to 40000.
        :param min_size_concept: Size of the internal prediction buffer and the minimum
            stable concept length retained for recovery after warnings. Defaults to
            7000.
        :param warning_limit: Number of consecutive warning instances allowed before
            warning is promoted to drift. Defaults to 1400.
        """

        cli = [
            f"-n {min_n_instances}",
            f"-w {warning_level}",
            f"-o {drift_level}",
            f"-x {max_size_concept}",
            f"-y {min_size_concept}",
            f"-z {warning_limit}",
        ]
        super().__init__(cli=" ".join(cli))
