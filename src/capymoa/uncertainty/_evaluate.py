from collections.abc import Mapping

from tqdm import tqdm

from capymoa.base import PredictionIntervalLearner
from capymoa.evaluation._loop import (
    _LoopOutput,
    _prequential_loop,
    _progress_label,
    _require_mapping,
    _require_single,
    _run_info,
    _supervised_step,
    _windows,
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
    window_size: int | None,
) -> PredictionIntervalResults:
    metrics = cumulative.metrics_dict()
    frame = _windows(windowed, [*_POINT, *_INTERVAL]) if windowed else None
    return PredictionIntervalResults(
        **_run_info(name, stream, out, window_size, frame),
        **{key: float(metrics[key]) for key in [*_POINT, *_INTERVAL]},
    )  # type: ignore[typeddict-item]


def evaluate_prediction_intervals(
    stream: Stream,
    learners: Mapping[str, PredictionIntervalLearner],
    max_instances: int | None = None,
    window_size: int | None = 1000,
    store_predictions: bool = False,
    store_y: bool = False,
    restart_stream: bool = True,
    progress_bar: bool | tqdm = False,
) -> dict[str, PredictionIntervalResults]:
    """Test-then-train many prediction interval learners on one pass over a stream.

    The learners get the same instances in turn, so the stream is read once.
    The training and testing of the learners is interleaved, so ``wallclock``
    and ``cpu_time`` are for the whole pass and are the same for all the
    learners. Do not use them to compare the speed of learners.

    ``y_pred`` has one row per instance with the lower bound, the point
    prediction and the upper bound.

    :param stream: The stream to evaluate on. Restarted if ``restart_stream``.
    :param max_instances: Number of instances to evaluate. If ``None``, go on
        until the stream ends.
    :param window_size: Number of instances in a window of ``windowed``. If
        ``None``, there are no windowed results.
    :param store_predictions: Keep the predictions in ``y_pred``.
    :param store_y: Keep the ground truth in ``y_true``.
    :param restart_stream: If ``False``, continue from the current position in the
        stream.
    :param progress_bar: Enable, disable, or give a progress bar.
    :param learners: The learners to evaluate, by name. The name is the ``learner``
        key of the result.
    :return: The results by name.
    """
    runs = _require_mapping(learners, "evaluate_prediction_interval")
    if restart_stream:
        stream.restart()
    schema = stream.get_schema()
    if not schema.is_regression():
        raise ValueError("The stream is not a regression stream.")

    cumulative = {n: PredictionIntervalEvaluator(schema) for n in runs}
    windowed = {
        n: None
        if window_size is None
        else PredictionIntervalWindowedEvaluator(schema, window_size)
        for n in runs
    }
    outs = _prequential_loop(
        stream,
        {n: _supervised_step(one) for n, one in runs.items()},
        cumulative,
        windowed,
        max_instances=max_instances,
        window_size=window_size,
        store_y=store_y,
        store_predictions=store_predictions,
        progress_bar=progress_bar,
        progress_label=_progress_label("Eval", runs, stream),
    )
    return {
        n: _results(n, stream, outs[n], cumulative[n], windowed[n], window_size)
        for n in runs
    }


def evaluate_prediction_interval(
    stream: Stream,
    learner: PredictionIntervalLearner,
    max_instances: int | None = None,
    window_size: int | None = 1000,
    store_predictions: bool = False,
    store_y: bool = False,
    restart_stream: bool = True,
    progress_bar: bool | tqdm = False,
) -> PredictionIntervalResults:
    """Test-then-train a prediction interval learner on a stream.

    >>> from capymoa.uncertainty import MVE, evaluate_prediction_interval
    >>> from capymoa.datasets import FriedTiny
    >>> stream = FriedTiny()
    >>> results = evaluate_prediction_interval(
    ...     stream, MVE(stream.get_schema()), max_instances=1000
    ... )
    >>> print(f"{results['coverage']:.1f}")
    97.8

    ``y_pred`` has one row per instance with the lower bound, the point
    prediction and the upper bound. There is no Java loop. To compare learners
    on one pass over the stream use :func:`evaluate_prediction_intervals`.

    :param stream: The stream to evaluate on. Restarted if ``restart_stream``.
    :param max_instances: Number of instances to evaluate. If ``None``, go on
        until the stream ends.
    :param window_size: Number of instances in a window of ``windowed``. If
        ``None``, there are no windowed results.
    :param store_predictions: Keep the predictions in ``y_pred``.
    :param store_y: Keep the ground truth in ``y_true``.
    :param restart_stream: If ``False``, continue from the current position in the
        stream.
    :param progress_bar: Enable, disable, or give a progress bar.
    :param learner: The learner to evaluate.
    :return: The results.
    """
    _require_single(learner, "evaluate_prediction_intervals")
    name = str(learner)
    return evaluate_prediction_intervals(
        stream,
        {name: learner},
        max_instances=max_instances,
        window_size=window_size,
        store_predictions=store_predictions,
        store_y=store_y,
        restart_stream=restart_stream,
        progress_bar=progress_bar,
    )[name]
