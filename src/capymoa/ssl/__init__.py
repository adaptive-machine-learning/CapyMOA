"""Semi-supervised learning.

Semi-supervised learning trains models using a mix of labeled and unlabeled
instances. In data stream learning, labels are often scarce or delayed, so
learners must exploit the abundant unlabeled data to improve predictions.
"""

from capymoa._optional import lazy_torch_attrs

from ._evaluate import evaluate_ssl
from ._results import SSLResults
from ._sleade import SLEADE

__all__ = ["OSNN", "SLEADE", "SSLResults", "evaluate_ssl"]


#: Names that need PyTorch. Imported on first access so ``import capymoa`` stays
#: torch-free -- see :mod:`capymoa._optional`.
_LAZY = {
    "OSNN": "._osnn",
}

__getattr__, __dir__ = lazy_torch_attrs(__name__, _LAZY, "OSNN", __all__)
