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

    _moa_detector_type = _PageHinkleyDM

    def __init__(
        self,
        min_n_instances: int = 30,
        delta: float = 0.005,
        lambda_: float = 50.0,
        alpha: float = 0.9999,
    ):
        """Create a Page-Hinkley drift detector.

        :param min_n_instances: Minimum number of instances to observe before change
            detection is enabled. Defaults to 30.
        :param delta: Slack term subtracted at each update of the running statistic.
            Larger values make the detector less sensitive to small shifts. Defaults to
            0.005.
        :param lambda_: Detection threshold for the Page-Hinkley statistic; once
            exceeded, a change is reported. Defaults to 50.0.
        :param alpha: Forgetting factor applied to the previous statistic. Values closer
            to 1.0 retain longer history and typically react more slowly to abrupt
            changes. Defaults to 0.9999.
        """
        cli = f"-n {min_n_instances} -d {delta} -l {lambda_} -a {alpha}"
        super().__init__(cli=cli)
