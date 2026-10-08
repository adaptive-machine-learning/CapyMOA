from collections.abc import Mapping
from typing import Any, overload

from tqdm import tqdm

from capymoa.base import PredictionIntervalLearner
from capymoa.evaluation._loop import (
    _LoopOutput,
    _prequential_loop,
    _Run,
    _run_info,
)
from capymoa.stream import Stream
from capymoa.uncertainty._results import PredictionIntervalResults
from capymoa.uncertainty.evaluate import (
    _INTERVAL,
    PredictionIntervalEvaluator,
    PredictionIntervalWindowedEvaluator,
)

_POINT = ["mae", "rmse", "rmae", "r2", "adjusted_r2"]


def _results(
    name: str,
    stream: Stream,
    out: _LoopOutput,
    cumulative: PredictionIntervalEvaluator,
    windowed: PredictionIntervalWindowedEvaluator | None,
) -> PredictionIntervalResults:
    metrics = cumulative.metrics_dict()
    return PredictionIntervalResults(
        **_run_info(name, stream, out, windowed, [*_POINT, *_INTERVAL]),
        **{key: float(metrics[key]) for key in [*_POINT, *_INTERVAL]},
    )  # type: ignore[typeddict-item]


@overload
def evaluate_prediction_interval(
    stream: Stream, learner: PredictionIntervalLearner, **kwargs: Any
) -> PredictionIntervalResults: ...
@overload
def evaluate_prediction_interval(
    stream: Stream, learner: Mapping[str, PredictionIntervalLearner], **kwargs: Any
) -> dict[str, PredictionIntervalResults]: ...
def evaluate_prediction_interval(
    stream: Stream,
    learner: PredictionIntervalLearner | Mapping[str, PredictionIntervalLearner],
    max_instances: int | None = None,
    window_size: int | None = 1000,
    store_predictions: bool = False,
    store_y: bool = False,
    restart_stream: bool = True,
    progress_bar: bool | tqdm = False,
) -> PredictionIntervalResults | dict[str, PredictionIntervalResults]:
    """Test-then-train a prediction interval learner, or many on one pass over a stream.

    Each instance is first used to test the learner, then to train it.

    >>> from capymoa.uncertainty import MVE, evaluate_prediction_interval
    >>> from capymoa.datasets import FriedTiny
    >>> stream = FriedTiny()
    >>> results = evaluate_prediction_interval(
    ...     stream, MVE(stream.get_schema()), max_instances=1000
    ... )
    >>> print(f"{results['coverage']:.1f}")
    97.8

    Give a mapping of names to learners to get a dict of results by name. The
    learners get the same instances in turn, so the stream is read once. This
    is useful when reading the stream is slow. The learners are interleaved, so
    ``wallclock`` and ``cpu_time`` are for the whole pass and are the same for
    all the learners. Do not use them to compare the speed of learners.

    ``y_pred`` has one row per instance with the lower bound, the point
    prediction and the upper bound. There is no Java loop.

    :param stream: The stream to evaluate on. Restarted if ``restart_stream``.
    :param learner: The learner to evaluate, or a mapping of names to learners.
        The name is the ``learner`` key of each result.
    :param max_instances: Number of instances to evaluate. If ``None``, go on
        until the stream ends.
    :param window_size: Number of instances in a window of ``windowed``. If
        ``None``, there are no windowed results.
    :param store_predictions: Keep the predictions in ``y_pred``.
    :param store_y: Keep the ground truth in ``y_true``.
    :param restart_stream: If ``False``, continue from the current position in the
        stream.
    :param progress_bar: Enable, disable, or give a progress bar.
    :return: The results, or a dict of results by name for a mapping.
    """
    many = isinstance(learner, Mapping)
    if many and not learner:
        raise ValueError("No learners to evaluate.")
    if restart_stream:
        stream.restart()
    schema = stream.get_schema()
    if not schema.is_regression():
        raise ValueError("The stream is not a regression stream.")

    learners = dict(learner) if many else {str(learner): learner}
    runs = {}
    for name, each in learners.items():
        windowed = None
        if window_size is not None:
            windowed = PredictionIntervalWindowedEvaluator(schema, window_size)
        runs[name] = _Run(
            each,
            PredictionIntervalEvaluator(schema),
            windowed,
            store_y=store_y,
            store_predictions=store_predictions,
        )
    outs = _prequential_loop(
        stream,
        runs,
        max_instances=max_instances,
        progress_bar=progress_bar,
    )
    results = {
        n: _results(n, stream, out, runs[n].cumulative, runs[n].windowed)
        for n, out in outs.items()
    }
    return results if many else next(iter(results.values()))
