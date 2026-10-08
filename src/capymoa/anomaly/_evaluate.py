from collections.abc import Mapping
from typing import Any

from tqdm import tqdm

from capymoa.anomaly._results import AnomalyResults
from capymoa.anomaly.evaluate import (
    AnomalyDetectionEvaluator,
    AnomalyDetectionWindowedEvaluator,
)
from capymoa.base import AnomalyDetector
from capymoa.evaluation._loop import (
    _is_fast_mode_compilable,
    _LoopOutput,
    _prequential_loop,
    _prequential_loop_fast,
    _progress_label,
    _require_mapping,
    _require_single,
    _Run,
    _run_info,
)
from capymoa.stream import Stream

_METRICS = ["auc", "s_auc"]


class _AnomalyRun(_Run):
    """Test-then-train an anomaly detector. It scores each instance, then trains."""

    def test_then_train(self, batch) -> tuple[list[Any], list[Any]]:
        learner = self.learner
        y_true, y_pred = [], []
        for instance in batch:
            y_pred.append(learner.score_instance(instance))
            y_true.append(instance.y_index)
            learner.train(instance)
        return y_true, y_pred


def _anomaly_results(
    name: str,
    stream: Stream,
    out: _LoopOutput,
    cumulative: AnomalyDetectionEvaluator,
    windowed: AnomalyDetectionWindowedEvaluator | None,
) -> AnomalyResults:
    metrics = cumulative.metrics_dict()
    return AnomalyResults(
        **_run_info(name, stream, out, windowed, _METRICS),
        **{key: float(metrics[key]) for key in _METRICS},
    )  # type: ignore[typeddict-item]


def evaluate_anomaly_detectors(
    stream: Stream,
    learners: Mapping[str, AnomalyDetector],
    max_instances: int | None = None,
    window_size: int | None = 1000,
    store_predictions: bool = False,
    store_y: bool = False,
    restart_stream: bool = True,
    progress_bar: bool | tqdm = False,
) -> dict[str, AnomalyResults]:
    """Test-then-train many anomaly detectors on one pass over a stream.

    The learners get the same instances in turn, so the stream is read once.
    The training and testing of the learners is interleaved, so ``wallclock``
    and ``cpu_time`` are for the whole pass and are the same for all the
    learners. Do not use them to compare the speed of learners.

    It does not use the Java loop. ``y_pred`` holds the anomaly scores.

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
    :param learners: The detectors to evaluate, by name. The name is the ``learner``
        key of the result.
    :return: The results by name.
    """
    learners = _require_mapping(learners, "evaluate_anomaly")
    for one in learners.values():
        if not isinstance(one, AnomalyDetector):
            raise TypeError("The learner is not an AnomalyDetector")
    if restart_stream:
        stream.restart()
    schema = stream.get_schema()

    runs = {
        n: _AnomalyRun(
            one,
            AnomalyDetectionEvaluator(schema=schema),
            None
            if window_size is None
            else AnomalyDetectionWindowedEvaluator(
                schema=schema, window_size=window_size
            ),
            store_y=store_y,
            store_predictions=store_predictions,
        )
        for n, one in learners.items()
    }
    outs = _prequential_loop(
        stream,
        runs,
        max_instances=max_instances,
        progress_bar=progress_bar,
        progress_label=_progress_label("AD Eval", learners, stream),
    )
    return {
        n: _anomaly_results(n, stream, out, runs[n].cumulative, runs[n].windowed)
        for n, out in outs.items()
    }


def evaluate_anomaly(
    stream: Stream,
    learner: AnomalyDetector,
    max_instances: int | None = None,
    window_size: int | None = 1000,
    store_predictions: bool = False,
    store_y: bool = False,
    optimise: bool = True,
    restart_stream: bool = True,
    progress_bar: bool | tqdm = False,
) -> AnomalyResults:
    """Test-then-train an anomaly detector on a stream.

    The detector scores each instance, then trains on it. ``y_pred`` holds the
    anomaly scores.

    >>> from capymoa.anomaly import HalfSpaceTrees, evaluate_anomaly
    >>> from capymoa.datasets import ElectricityTiny
    >>> stream = ElectricityTiny()
    >>> results = evaluate_anomaly(
    ...     stream, HalfSpaceTrees(stream.get_schema()), max_instances=1000
    ... )
    >>> print(f"{results['auc']:.2f}")
    0.38

    To compare detectors on one pass over the stream use
    :func:`evaluate_anomaly_detectors`.

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
    :param learner: The detector to evaluate.
    :param optimise: Use the Java loop in MOA if the detector allows it. Needs a
        ``window_size``. The Java loop has no progress bar.
    :return: The results.
    """
    _require_single(learner, "evaluate_anomaly_detectors")
    if not isinstance(learner, AnomalyDetector):
        raise TypeError("The learner is not an AnomalyDetector")
    if restart_stream:
        stream.restart()
    schema = stream.get_schema()
    name = str(learner)
    if window_size is not None and _is_fast_mode_compilable(stream, learner, optimise):
        run = _Run(
            learner,
            AnomalyDetectionEvaluator(schema=schema),
            AnomalyDetectionWindowedEvaluator(schema=schema, window_size=window_size),
            store_y=store_y,
            store_predictions=store_predictions,
        )
        out = _prequential_loop_fast(stream, run, max_instances=max_instances)
        return _anomaly_results(name, stream, out, run.cumulative, run.windowed)
    return evaluate_anomaly_detectors(
        stream,
        {name: learner},
        max_instances=max_instances,
        window_size=window_size,
        store_predictions=store_predictions,
        store_y=store_y,
        restart_stream=False,
        progress_bar=progress_bar,
    )[name]
