from typing import Optional

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

    def __init__(
        self,
        min_n_instances: int = 30,
        warning_level: float = 2.0,
        out_control_level: float = 3.0,
        CLI: Optional[str] = None,
    ):
        if CLI is None:
            CLI = f"-n {min_n_instances} -w {warning_level} -o {out_control_level}"

        super().__init__(moa_detector=_DDM(), CLI=CLI)

        self.min_n_instances = min_n_instances
        self.warning_level = warning_level
        self.out_control_level = out_control_level
        self.get_params()
