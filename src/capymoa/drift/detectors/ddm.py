from capymoa.drift.base_detector import MOADriftDetector

from moa.classifiers.core.driftdetection import DDM as _DDM


class DDM(MOADriftDetector):
    """Drift-Detection-Method (DDM) Drift Detector

    Example:
    --------

    >>> import numpy as np
    >>> from capymoa.drift.detectors import DDM
    >>> np.random.seed(0)
    >>>
    >>> detector = DDM()
    >>>
    >>> data_stream = np.random.randint(2, size=2000)
    >>> for i in range(999, 2000):
    ...     data_stream[i] = np.random.randint(4, high=8)
    >>>
    >>> for i in range(2000):
    ...     detector.add_element(data_stream[i])
    ...     if detector.detected_change():
    ...         print('Change detected in data: ' + str(data_stream[i]) + ' - at index: ' + str(i))
    Change detected in data: 4 - at index: 1005

    Reference:
    ----------

    Gama, Joao, et al. "Learning with drift detection." Advances in Artificial
    Intelligence–SBIA 2004: 17th Brazilian Symposium on Artificial Intelligence,
    Sao Luis, Maranhao, Brazil, September 29-Ocotber 1, 2004.

    """

    _moa_detector_type = _DDM

    def __init__(
        self,
        min_n_instances: int = 30,
        warning_level: float = 2.0,
        out_control_level: float = 3.0,
    ):
        """Create a DDM drift detector.

        :param min_n_instances: Minimum number of instances to observe before change
            detection is enabled. Defaults to 30.
        :param warning_level: Multiplier applied to the minimum error estimate for
            entering the warning zone. Defaults to 2.0.
        :param out_control_level: Multiplier applied to the minimum error estimate for
            reporting a concept change. Defaults to 3.0.
        """
        cli = f"-n {min_n_instances} -w {warning_level} -o {out_control_level}"
        super().__init__(cli=cli)
