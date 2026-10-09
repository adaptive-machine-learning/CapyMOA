from collections.abc import Mapping
from typing import Any, overload

from tqdm import tqdm
from typing_extensions import override

from capymoa.anomaly.evaluate import (
    AnomalyDetectionEvaluator,
    AnomalyDetectionWindowedEvaluator,
    AnomalyResults,
    AnomalyWindows,
)
from capymoa.base import AnomalyDetector
from capymoa.evaluation._loop import (
    _LoopOutput,
    _prequential_loop,
    _prequential_loop_fast,
    _results_body,
    _Run,
    _use_java_loop,
)
from capymoa.stream import Stream


class _AnomalyRun(_Run):
    """Test-then-train an anomaly detector. It scores each instance, then trains."""

    @override
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
    body = _results_body(name, stream, out, cumulative, windowed, AnomalyWindows)
    return AnomalyResults(**body)  # type: ignore[typeddict-item]


@overload
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
) -> AnomalyResults: ...
@overload
def evaluate_anomaly(
    stream: Stream,
    learner: Mapping[str, AnomalyDetector],
    max_instances: int | None = None,
    window_size: int | None = 1000,
    store_predictions: bool = False,
    store_y: bool = False,
    optimise: bool = True,
    restart_stream: bool = True,
    progress_bar: bool | tqdm = False,
) -> dict[str, AnomalyResults]: ...
def evaluate_anomaly(
    stream: Stream,
    learner: AnomalyDetector | Mapping[str, AnomalyDetector],
    max_instances: int | None = None,
    window_size: int | None = 1000,
    store_predictions: bool = False,
    store_y: bool = False,
    optimise: bool = True,
    restart_stream: bool = True,
    progress_bar: bool | tqdm = False,
) -> AnomalyResults | dict[str, AnomalyResults]:
    """Test-then-train an anomaly detector, or many on one pass over a stream.

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

    Give a mapping of names to detectors to get a dict of results by name. The
    detectors get the same instances in turn, so the stream is read once. A
    mapping does not use the Java loop. The training and testing of the
    detectors is interleaved, so ``wallclock`` and ``cpu_time`` are for the
    whole pass and are the same for all the detectors. Do not use them to
    compare the speed of detectors.

    :param stream: The stream to evaluate on. Restarted if ``restart_stream``.
    :param learner: The detector to evaluate, or a mapping of names to detectors.
        The name is the ``learner`` key of each result.
    :param max_instances: Number of instances to evaluate. If ``None``, go on
        until the stream ends.
    :param window_size: Number of instances in a window of ``windowed``. If
        ``None``, there are no windowed results.
    :param store_predictions: Keep the predictions in ``y_pred``.
    :param store_y: Keep the ground truth in ``y_true``.
    :param optimise: Use the Java loop in MOA if the detector allows it. Needs a
        ``window_size``. The Java loop has no progress bar. Only used for one
        detector.
    :param restart_stream: If ``False``, continue from the current position in the
        stream.
    :param progress_bar: Enable, disable, or give a progress bar.
    :return: The results, or a dict of results by name for a mapping.
    """
    many = isinstance(learner, Mapping)
    if many and not learner:
        raise ValueError("No learners to evaluate.")
    each_learner = learner.values() if many else [learner]
    for one in each_learner:
        if not isinstance(one, AnomalyDetector):
            raise TypeError("The learner is not an AnomalyDetector")
    if restart_stream:
        stream.restart()
    schema = stream.get_schema()

    if not many and _use_java_loop(
        stream, learner, optimise=optimise, window_size=window_size
    ):
        run = _Run(
            learner,
            AnomalyDetectionEvaluator(schema=schema),
            AnomalyDetectionWindowedEvaluator(schema=schema, window_size=window_size),
            store_y=store_y,
            store_predictions=store_predictions,
        )
        out = _prequential_loop_fast(stream, run, max_instances=max_instances)
        return _anomaly_results(str(learner), stream, out, run.cumulative, run.windowed)

    learners = dict(learner) if many else {str(learner): learner}
    runs = {}
    for name, each in learners.items():
        windowed = None
        if window_size is not None:
            windowed = AnomalyDetectionWindowedEvaluator(
                schema=schema, window_size=window_size
            )
        runs[name] = _AnomalyRun(
            each,
            AnomalyDetectionEvaluator(schema=schema),
            windowed,
            store_y=store_y,
            store_predictions=store_predictions,
        )
    outs = _prequential_loop(
        stream,
        runs,
        max_instances=max_instances,
        progress_bar=progress_bar,
        progress_prefix="AD Eval",
    )
    results = {
        n: _anomaly_results(n, stream, out, runs[n].cumulative, runs[n].windowed)
        for n, out in outs.items()
    }
    return results if many else next(iter(results.values()))
