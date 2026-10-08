from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, overload

from tqdm import tqdm

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
@overload
def prequential_evaluation(
    stream: Stream, learner: Mapping[str, Classifier], **kwargs: Any
) -> "dict[str, ClassifierResults]": ...
@overload
def prequential_evaluation(
    stream: Stream, learner: Mapping[str, Regressor], **kwargs: Any
) -> "dict[str, RegressorResults]": ...
def prequential_evaluation(
    stream: Stream,
    learner: Any,
    max_instances: int | None = None,
    window_size: int | None = 1000,
    store_predictions: bool = False,
    store_y: bool = False,
    optimise: bool = True,
    restart_stream: bool = True,
    progress_bar: bool | tqdm = False,
    batch_size: int = 1,
):
    """Test-then-train a learner on a stream, whatever its type.

    Calls the ``evaluate_*`` function of the learner's domain:
    :func:`~capymoa.classifier.evaluate_classifier`,
    :func:`~capymoa.regressor.evaluate_regressor`,
    :func:`~capymoa.uncertainty.evaluate_prediction_interval` or
    :func:`~capymoa.anomaly.evaluate_anomaly`. A mapping of names to learners is
    passed through as is, and the result is a dict of results by name. Use
    those functions to see the parameters and the results of a domain.

    :param learner: A learner, or a mapping of names to learners of one domain.
    :return: The results of the domain, or a dict of results by name for a
        mapping.
    :raises TypeError: If the learner is not a known type.
    :raises ValueError: If the mapping of learners is empty.
    """
    if isinstance(learner, Mapping) and not learner:
        raise ValueError("No learners to evaluate.")
    sample = next(iter(learner.values())) if isinstance(learner, Mapping) else learner
    kwargs: dict[str, Any] = {
        "max_instances": max_instances,
        "window_size": window_size,
        "store_predictions": store_predictions,
        "store_y": store_y,
        "restart_stream": restart_stream,
        "progress_bar": progress_bar,
    }
    if isinstance(sample, PredictionIntervalLearner):
        from capymoa.uncertainty import evaluate_prediction_interval

        function = evaluate_prediction_interval
    elif isinstance(sample, Classifier):
        from capymoa.classifier import evaluate_classifier

        function = evaluate_classifier
        kwargs["batch_size"] = batch_size
        kwargs["optimise"] = optimise
    elif isinstance(sample, Regressor):
        from capymoa.regressor import evaluate_regressor

        function = evaluate_regressor
        kwargs["batch_size"] = batch_size
        kwargs["optimise"] = optimise
    elif isinstance(sample, AnomalyDetector):
        from capymoa.anomaly import evaluate_anomaly

        function = evaluate_anomaly
        kwargs["optimise"] = optimise
    else:
        raise TypeError(f"Cannot evaluate a learner of type {type(sample).__name__}")
    return function(stream, learner, **kwargs)
