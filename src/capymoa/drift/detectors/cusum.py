from capymoa.drift.base_detector import MOADriftDetector

from moa.classifiers.core.driftdetection import CusumDM as _CusumDM


class CUSUM(MOADriftDetector):
    """CUSUM Drift Detector

    Example usages:

    >>> import numpy as np
    >>> np.random.seed(0)
    >>> from capymoa.drift.detectors import CUSUM
    >>>
    >>> detector = CUSUM(delta=0.005, lambda_=60)
    >>>
    >>> data_stream = np.random.randint(2, size=2000)
    >>> for i in range(999, 2000):
    ...     data_stream[i] = np.random.randint(4, high=8)
    >>>
    >>> for i in range(2000):
    ...     detector.add_element(data_stream[i])
    ...     if detector.detected_change():
    ...         print('Change detected in data: ' + str(data_stream[i]) + ' - at index: ' + str(i))
    Change detected in data: 6 - at index: 1011
    Change detected in data: 7 - at index: 1556

    """

    _moa_detector_type = _CusumDM

    def __init__(
        self,
        min_n_instances: int = 30,
        delta: float = 0.005,
        lambda_: float = 50,
    ):
        """Create a CUSUM drift detector.

        :param min_n_instances: Minimum number of instances to observe before change
            detection is enabled. Defaults to 30.
        :param delta: CUSUM slack parameter subtracted from the running sum at each
            step. Larger values make the detector less sensitive to small shifts.
            Defaults to 0.005.
        :param lambda_: Threshold for the cumulative sum; once the running sum exceeds
            this value, a change is reported. Defaults to 50.
        """
        cli = f"-n {min_n_instances} -d {delta} -l {lambda_}"
        super().__init__(cli=cli)
