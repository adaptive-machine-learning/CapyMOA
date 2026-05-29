from capymoa.drift.base_detector import MOADriftDetector

from moa.classifiers.core.driftdetection import (
    SEEDChangeDetector as _SEEDChangeDetector,
)


class SEED(MOADriftDetector):
    """Seed Drift Detector

    Example:
    --------

    >>> import numpy as np
    >>> from capymoa.drift.detectors import SEED
    >>> np.random.seed(0)
    >>>
    >>> detector = SEED()
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
    Change detected in data: 6 - at index: 1343

    Reference:
    ----------

    Huang, David Tse Jung, et al. "Detecting volatility shift in data streams."
    2014 IEEE International Conference on Data Mining. IEEE, 2014.

    """

    _moa_detector_type = _SEEDChangeDetector

    def __init__(
        self,
        delta: float = 0.05,
        block_size: int = 32,
        epsilon_prime: float = 0.01,
        alpha: float = 0.8,
        compress_term: int = 75,
    ):
        """Create a SEED drift detector.

        :param delta: Confidence parameter used in the ADWIN-style cut bound inside
            SEED. Smaller values make drift declarations more conservative. Defaults to
            0.05.
        :param block_size: Number of instances per block before drift checks are
            attempted. Defaults to 32.
        :param epsilon_prime: Base homogeneity tolerance used by SEED block compression.
            Defaults to 0.01.
        :param alpha: Growth parameter used with ``epsilon_prime`` during compression;
            larger values increase compression tolerance. Defaults to 0.8.
        :param compress_term: Compression interval controlling how often fixed-term
            block compression is attempted. Defaults to 75.
        """
        cli = [
            f"-d {delta}",
            f"-b {block_size}",
            f"-e {epsilon_prime}",
            f"-a {alpha}",
            f"-c {compress_term}",
        ]
        super().__init__(cli=" ".join(cli))
