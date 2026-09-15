"""Concept and data drift detectors.

This module exports both families.

* **Concept drift** (``REQUIRES_FIT = False``): feed a scalar, usually
  a prediction error, to :meth:`~capymoa.drift.base_detector.BaseDriftDetector.add_element`.
* **Data drift** (``REQUIRES_FIT = True``): compare input features to a
  reference with :meth:`~capymoa.drift.detectors.BaseDataDriftDetector.fit`
  or ``auto_fit_samples``.

:class:`BaseDataDriftDetector` and :class:`DataDriftResult` are helpers,
not detectors to pick.

Concept drift
-------------

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Detector
     - Typical input
   * - :class:`ABCD`
     - Feature vector, or a scalar error
   * - :class:`ADWIN`
     - Scalar
   * - :class:`CUSUM`
     - Scalar
   * - :class:`DDM`
     - Scalar
   * - :class:`EWMAChart`
     - Scalar
   * - :class:`GeometricMovingAverage`
     - Scalar
   * - :class:`HDDMAverage`
     - Scalar
   * - :class:`HDDMWeighted`
     - Scalar
   * - :class:`OPTWIN`
     - Scalar
   * - :class:`PageHinkley`
     - Scalar
   * - :class:`RDDM`
     - Scalar
   * - :class:`SEED`
     - Scalar
   * - :class:`STEPD`
     - Scalar
   * - :class:`STUDD`
     - Feature vector and model prediction

Data drift
----------

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Detector
     - Typical input
   * - :class:`AndersonDarling`
     - Feature vector (numeric)
   * - :class:`BNDM`
     - Feature vector (numeric)
   * - :class:`ChiSquare`
     - Feature vector (categorical)
   * - :class:`CramerVonMises`
     - Feature vector (numeric)
   * - :class:`D3`
     - Feature vector (numeric)
   * - :class:`EnergyDistance`
     - Feature vector (numeric)
   * - :class:`Hellinger`
     - Feature vector (numeric)
   * - :class:`IBDD`
     - Feature vector (numeric)
   * - :class:`JensenShannon`
     - Feature vector (numeric)
   * - :class:`KLDivergence`
     - Feature vector (numeric)
   * - :class:`KolmogorovSmirnov`
     - Feature vector (numeric)
   * - :class:`MMD`
     - Feature vector (numeric)
   * - :class:`PSI`
     - Feature vector (numeric)
   * - :class:`Wasserstein`
     - Feature vector (numeric)
"""

from .abcd import ABCD
from .adwin import ADWIN
from .cusum import CUSUM
from .ddm import DDM
from .ewma_chart import EWMAChart
from .geometric_ma import GeometricMovingAverage
from .hddm_a import HDDMAverage
from .hddm_w import HDDMWeighted
from .optwin import OPTWIN
from .page_hinkley import PageHinkley
from .rddm import RDDM
from .seed import SEED
from .stepd import STEPD
from .studd import STUDD
from .data_drift import (
    AndersonDarling,
    BaseDataDriftDetector,
    BNDM,
    ChiSquare,
    CramerVonMises,
    D3,
    DataDriftResult,
    EnergyDistance,
    Hellinger,
    IBDD,
    JensenShannon,
    KLDivergence,
    KolmogorovSmirnov,
    MMD,
    PSI,
    Wasserstein,
)

__all__ = [
    "ABCD",
    "ADWIN",
    "CUSUM",
    "DDM",
    "EWMAChart",
    "GeometricMovingAverage",
    "HDDMAverage",
    "HDDMWeighted",
    "OPTWIN",
    "PageHinkley",
    "RDDM",
    "SEED",
    "STEPD",
    "STUDD",
    "AndersonDarling",
    "BaseDataDriftDetector",
    "BNDM",
    "ChiSquare",
    "CramerVonMises",
    "D3",
    "DataDriftResult",
    "EnergyDistance",
    "Hellinger",
    "IBDD",
    "JensenShannon",
    "KLDivergence",
    "KolmogorovSmirnov",
    "MMD",
    "PSI",
    "Wasserstein",
]
