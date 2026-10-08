from collections.abc import Mapping
from typing import Any, overload

from tqdm import tqdm

from capymoa.base import Regressor
from capymoa.evaluation._loop import (
    _LoopOutput,
    _prequential_loop,
    _prequential_loop_fast,
    _results_body,
    _Run,
    _use_java_loop,
)
from capymoa.regressor._results import RegressorResults, RegressorWindows
from capymoa.regressor.evaluate import RegressionEvaluator, RegressionWindowedEvaluator
from capymoa.stream import Stream


def _regressor_results(
    name: str,
    stream: Stream,
    out: _LoopOutput,
    cumulative: RegressionEvaluator,
    windowed: RegressionWindowedEvaluator | None,
) -> RegressorResults:
    body = _results_body(name, stream, out, cumulative, windowed, RegressorWindows)
    return RegressorResults(**body)  # type: ignore[typeddict-item]


@overload
def evaluate_regressor(
    stream: Stream, learner: Regressor, **kwargs: Any
) -> RegressorResults: ...
@overload
def evaluate_regressor(
    stream: Stream, learner: Mapping[str, Regressor], **kwargs: Any
) -> dict[str, RegressorResults]: ...
def evaluate_regressor(
    stream: Stream,
    learner: Regressor | Mapping[str, Regressor],
    max_instances: int | None = None,
    window_size: int | None = 1000,
    store_predictions: bool = False,
    store_y: bool = False,
    optimise: bool = True,
    restart_stream: bool = True,
    progress_bar: bool | tqdm = False,
    batch_size: int = 1,
) -> RegressorResults | dict[str, RegressorResults]:
    """Test-then-train a regressor, or many on one pass over a stream.

    Each instance is first used to test the learner, then to train it.

    >>> from capymoa.regressor import FIMTDD, evaluate_regressor
    >>> from capymoa.datasets import FriedTiny
    >>> stream = FriedTiny()
    >>> results = evaluate_regressor(
    ...     stream, FIMTDD(stream.get_schema()), max_instances=1000
    ... )
    >>> print(f"{results['rmse']:.2f}")
    7.36

    Give a mapping of names to learners to get a dict of results by name. The
    learners get the same instances in turn, so the stream is read once. This
    is useful when reading the stream is slow. A mapping does not use the Java
    loop. The learners are interleaved, so ``wallclock`` and ``cpu_time`` are
    for the whole pass and are the same for all the learners. Do not use them
    to compare the speed of learners.

    >>> from capymoa.regressor import TargetMean
    >>> learners = {
    ...     "fimtdd": FIMTDD(stream.get_schema()),
    ...     "mean": TargetMean(stream.get_schema()),
    ... }
    >>> results = evaluate_regressor(stream, learners, max_instances=1000)
    >>> list(results)
    ['fimtdd', 'mean']

    :param stream: The stream to evaluate on. Restarted if ``restart_stream``.
    :param learner: The learner to evaluate, or a mapping of names to learners.
        The name is the ``learner`` key of each result.
    :param max_instances: Number of instances to evaluate. If ``None``, go on
        until the stream ends.
    :param window_size: Number of instances in a window of ``windowed``. If
        ``None``, there are no windowed results.
    :param store_predictions: Keep the predictions in ``y_pred``.
    :param store_y: Keep the ground truth in ``y_true``.
    :param optimise: Use the Java loop in MOA if the learner allows it. Needs a
        ``window_size``. The Java loop has no progress bar. Only used for one
        learner.
    :param restart_stream: If ``False``, continue from the current position in the
        stream.
    :param progress_bar: Enable, disable, or give a progress bar.
    :param batch_size: Instances per mini-batch, for batch learners.
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

    if not many and _use_java_loop(
        stream,
        learner,
        optimise=optimise,
        window_size=window_size,
        batch_size=batch_size,
    ):
        run = _Run(
            learner,
            RegressionEvaluator(schema=schema),
            RegressionWindowedEvaluator(schema=schema, window_size=window_size),
            store_y=store_y,
            store_predictions=store_predictions,
        )
        out = _prequential_loop_fast(stream, run, max_instances=max_instances)
        return _regressor_results(
            str(learner), stream, out, run.cumulative, run.windowed
        )

    learners = dict(learner) if many else {str(learner): learner}
    runs = {}
    for name, each in learners.items():
        windowed = None
        if window_size is not None:
            windowed = RegressionWindowedEvaluator(
                schema=schema, window_size=window_size
            )
        runs[name] = _Run(
            each,
            RegressionEvaluator(schema=schema),
            windowed,
            store_y=store_y,
            store_predictions=store_predictions,
        )
    outs = _prequential_loop(
        stream,
        runs,
        max_instances=max_instances,
        progress_bar=progress_bar,
        batch_size=batch_size,
    )
    results = {
        n: _regressor_results(n, stream, out, runs[n].cumulative, runs[n].windowed)
        for n, out in outs.items()
    }
    return results if many else next(iter(results.values()))
