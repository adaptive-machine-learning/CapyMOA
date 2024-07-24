from typing import Optional

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

    def __init__(
        self,
        delta: float = 0.05,
        block_size: int = 32,
        epsilon_prime: float = 0.01,
        alpha: float = 0.8,
        compress_term: int = 75,
        CLI: Optional[str] = None,
    ):
        if CLI is None:
            CLI = (
                f"-d {delta} "
                f"-b {block_size} "
                f"-e {epsilon_prime} "
                f"-a {alpha} "
                f"-c {compress_term}"
            )

        super().__init__(moa_detector=_SEEDChangeDetector(), CLI=CLI)

        self.delta = delta
        self.block_size = block_size
        self.epsilon_prime = epsilon_prime
        self.alpha = alpha
        self.compress_term = compress_term
        self.get_params()
