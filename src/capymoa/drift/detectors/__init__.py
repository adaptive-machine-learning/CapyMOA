from capymoa._optional import lazy_torch_attrs
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
]


#: ABCD uses a torch autoencoder. Imported on first access so ``import capymoa``
#: stays torch-free -- see :mod:`capymoa._optional`.
_LAZY = {
    "ABCD": ".abcd",
}

__getattr__, __dir__ = lazy_torch_attrs(
    __name__, _LAZY, "the ABCD drift detector", __all__
)
