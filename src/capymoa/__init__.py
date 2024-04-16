from .prepare_jpype import _start_jpype
_start_jpype()
"""Whenever capymoa is imported, start jpype.
"""

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