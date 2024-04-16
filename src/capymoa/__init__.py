from .prepare_jpype import _start_jpype
# It is important that this is called before importing any other module
_start_jpype()
from .base import Regressor, Classifier, ClassifierSSL
from .stream.instance import Instance, LabeledInstance, RegressionInstance

__all__ = [
    "Regressor",
    "Classifier",
    "ClassifierSSL",
    "Instance",
    "LabeledInstance",
    "RegressionInstance",
]
