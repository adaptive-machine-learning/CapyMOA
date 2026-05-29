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

    _moa_detector_type = _STEPD

    def __init__(
        self,
        window_size: int = 30,
        alpha_drift: float = 0.003,
        alpha_warning: float = 0.05,
    ):
        """Create a STEPD drift detector.

        :param window_size: Size of the recent window used by STEPD to compare recent
            and older prediction proportions. Larger values smooth short-term noise but
            usually react more slowly. Defaults to 30.
        :param alpha_drift: Significance level for declaring drift. Smaller values make
            drift alarms more conservative; larger values make them easier to trigger.
            Defaults to 0.003.
        :param alpha_warning: Significance level for entering warning state. This is
            typically less strict than ``alpha_drift`` so warnings can appear before
            full drift. Defaults to 0.05.
        """
        cli = f"-r {window_size} -o {alpha_drift} -w {alpha_warning}"
        super().__init__(cli=cli)
