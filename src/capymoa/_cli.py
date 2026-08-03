"""Internal functions for generating CLI creation strings for MOA objects."""

from capymoa.base import MOAClassifier, MOARegressor, MOAPredictionIntervalLearner
from capymoa.drift.base_detector import MOADriftDetector
from capymoa.stream import MOAStream

from moa.streams import InstanceStream as _InstanceStream
from moa.options import AbstractOptionHandler as _AbstractOptionHandler
from moa.classifiers import AbstractClassifier as _AbstractClassifier
from moa.classifiers import Regressor as _AbstractRegressor
from moa.classifiers.core.driftdetection import (
    AbstractChangeDetector as _AbstractChangeDetector,
)
from moa.classifiers.predictioninterval import (
    PredictionIntervalLearner as _PredictionIntervalLearner,
)
from typing import Type


def cli_str(object: _AbstractOptionHandler, type_: Type[_AbstractOptionHandler]) -> str:
    """Return a CLI string for creating MOA objects.

    >>> from moa.classifiers.trees import HoeffdingTree
    >>> from moa.classifiers import AbstractClassifier
    >>> cli_str(HoeffdingTree(), AbstractClassifier)
    '(trees.HoeffdingTree)'

    :param object: MOA object to generate CLI string for.
    :param type_: Type of the MOA object, used to strip the fully qualified name to be
        more concise.
    :return: CLI string for creating the MOA object.
    """
    return f"({str(object.getCLICreationString(type_)).strip()})"


def cli_str_classifier(classifier: _AbstractClassifier | MOAClassifier) -> str:
    """Return a CLI string for creating a MOA classifier.

    >>> from moa.classifiers.trees import HoeffdingTree
    >>> cli_str_classifier(HoeffdingTree())
    '(trees.HoeffdingTree)'

    """
    if isinstance(classifier, MOAClassifier):
        return cli_str(classifier.moa_learner, _AbstractClassifier)
    elif isinstance(classifier, _AbstractClassifier):
        return cli_str(classifier, _AbstractClassifier)
    else:
        raise ValueError("Unknown Type")


def cli_str_regressor(regressor: _AbstractRegressor | MOARegressor) -> str:
    """Return a CLI string for creating a MOA regressor.

    >>> from moa.classifiers.trees import HoeffdingTree
    >>> cli_str_regressor(HoeffdingTree())
    '(trees.HoeffdingTree)'

    """
    if isinstance(regressor, MOARegressor):
        # TODO: type ignore should be removed if regressor.moa_learner is properly typed
        return cli_str(regressor.moa_learner, _AbstractRegressor)  # type: ignore
    elif isinstance(regressor, _AbstractRegressor):
        return cli_str(regressor, _AbstractRegressor)  # type: ignore
    elif isinstance(regressor, (_AbstractClassifier, MOAClassifier)):
        # Some regressor are just classifiers.
        return cli_str_classifier(regressor)
    else:
        raise ValueError("Unknown Type")


def cli_str_drift_detector(
    detector: _AbstractChangeDetector | MOADriftDetector,
) -> str:
    """Return a CLI string for creating a MOA drift detector.

    >>> from moa.classifiers.core.driftdetection import ADWINChangeDetector
    >>> cli_str_drift_detector(ADWINChangeDetector())
    '(ADWINChangeDetector)'

    """
    if isinstance(detector, MOADriftDetector):
        return cli_str(detector.moa_detector, _AbstractChangeDetector)
    elif isinstance(detector, _AbstractChangeDetector):
        return cli_str(detector, _AbstractChangeDetector)
    else:
        raise ValueError("Unknown Type")


def cli_str_prediction_interval(
    predictor: _PredictionIntervalLearner | MOAPredictionIntervalLearner,
) -> str:
    """Return a CLI string for creating a MOA prediction interval learner.

    >>> from moa.classifiers.predictioninterval import MVEPredictionInterval
    >>> cli_str_prediction_interval(MVEPredictionInterval())
    '(MVEPredictionInterval...

    """
    if isinstance(predictor, MOAPredictionIntervalLearner):
        return cli_str(predictor.moa_learner, _PredictionIntervalLearner)  # type: ignore
    elif isinstance(predictor, _PredictionIntervalLearner):
        return cli_str(predictor, _PredictionIntervalLearner)
    else:
        raise ValueError("Unknown Type")


def cli_str_stream(stream: MOAStream | _InstanceStream) -> str:
    """Return a CLI string for creating a MOA stream.

    >>> from moa.streams.generators import RandomRBFGenerator
    >>> cli_str_stream(RandomRBFGenerator())
    '(generators.RandomRBFGenerator)'

    """
    if isinstance(stream, MOAStream):
        return cli_str(stream.moa_stream, _InstanceStream)
    elif isinstance(stream, _InstanceStream):
        return cli_str(stream, _InstanceStream)
    else:
        raise ValueError("Unknown Type")
