from typing import TYPE_CHECKING, Any, overload

from capymoa.base import (
    AnomalyDetector,
    Classifier,
    PredictionIntervalLearner,
    Regressor,
)
from capymoa.stream import Stream

if TYPE_CHECKING:
    from capymoa.anomaly import AnomalyResults
    from capymoa.classifier import ClassifierResults
    from capymoa.regressor import RegressorResults
    from capymoa.uncertainty import PredictionIntervalResults


@overload
def prequential_evaluation(
    stream: Stream, learner: PredictionIntervalLearner, **kwargs: Any
) -> "PredictionIntervalResults": ...
@overload
def prequential_evaluation(
    stream: Stream, learner: Classifier, **kwargs: Any
) -> "ClassifierResults": ...
@overload
def prequential_evaluation(
    stream: Stream, learner: Regressor, **kwargs: Any
) -> "RegressorResults": ...
@overload
def prequential_evaluation(
    stream: Stream, learner: AnomalyDetector, **kwargs: Any
) -> "AnomalyResults": ...
def prequential_evaluation(stream: Stream, learner: Any, **kwargs: Any):
    """Test-then-train a learner on a stream, whatever its type.

    Calls the ``evaluate_*`` function of the learner's domain:
    :func:`~capymoa.classifier.evaluate_classifier`,
    :func:`~capymoa.regressor.evaluate_regressor`,
    :func:`~capymoa.uncertainty.evaluate_prediction_interval` or
    :func:`~capymoa.anomaly.evaluate_anomaly`. Keyword arguments go to that
    function. See it for the parameters and the results. To compare many
    learners, pass a mapping to the domain function.

    :raises TypeError: If the learner is not a known type.
    """
    if isinstance(learner, PredictionIntervalLearner):
        from capymoa.uncertainty import evaluate_prediction_interval

        function = evaluate_prediction_interval
    elif isinstance(learner, Classifier):
        from capymoa.classifier import evaluate_classifier

        function = evaluate_classifier
    elif isinstance(learner, Regressor):
        from capymoa.regressor import evaluate_regressor

        function = evaluate_regressor
    elif isinstance(learner, AnomalyDetector):
        from capymoa.anomaly import evaluate_anomaly

        function = evaluate_anomaly
    else:
        raise TypeError(f"Cannot evaluate a learner of type {type(learner).__name__}")
    return function(stream, learner, **kwargs)
