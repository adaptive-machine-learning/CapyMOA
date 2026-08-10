"""Anomaly detection.

Anomaly detection identifies instances that deviate substantially from normal
behavior. In data stream learning, the notion of normal behavior can evolve
over time, so detectors must adapt to concept drift while flagging outliers
in real time.
"""

from capymoa._optional import lazy_torch_attrs
from ._adaptive_isolation_forest import AdaptiveIsolationForest
from ._half_space_trees import HalfSpaceTrees
from ._iforest_asd import IForestASD
from ._online_isolation_forest import OnlineIsolationForest
from ._robust_random_cut_forest import RobustRandomCutForest
from ._stream_rhf import StreamRHF
from ._streaming_isolation_forest import StreamingIsolationForest
from ._loda import Loda
from ._rs_hash import RSHash

__all__ = [
    "AdaptiveIsolationForest",
    "Autoencoder",
    "HalfSpaceTrees",
    "Loda",
    "OnlineIsolationForest",
    "RSHash",
    "RobustRandomCutForest",
    "StreamRHF",
    "StreamingIsolationForest",
    "RobustRandomCutForest",
    "AdaptiveIsolationForest",
    "IForestASD",
    "Loda",
]


#: Names that need PyTorch. Imported on first access so ``import capymoa`` stays
#: torch-free -- see :mod:`capymoa._optional`.
_LAZY = {
    "Autoencoder": "._autoencoder",
}

__getattr__, __dir__ = lazy_torch_attrs(__name__, _LAZY, "Autoencoder", __all__)
