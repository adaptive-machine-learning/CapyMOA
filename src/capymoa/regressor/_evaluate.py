from collections.abc import Mapping

from tqdm import tqdm

from capymoa.base import Regressor
from capymoa.evaluation._loop import (
    _check_batch_size,
    _is_fast_mode_compilable,
    _LoopOutput,
    _prequential_loop,
    _prequential_loop_fast,
    _progress_label,
    _require_mapping,
    _require_single,
    _run_info,
    _supervised_step,
    _windows,
)
from capymoa.regressor._results import RegressorResults
from capymoa.regressor.evaluate import RegressionEvaluator, RegressionWindowedEvaluator
from capymoa.stream import Stream

_METRICS = ["mae", "rmse", "rmae", "r2", "adjusted_r2"]


def _regressor_results(
    name: str,
    stream: Stream,
    out: _LoopOutput,
    cumulative: RegressionEvaluator,
    windowed: RegressionWindowedEvaluator | None,
    window_size: int | None,
) -> RegressorResults:
    metrics = cumulative.metrics_dict()
    frame = _windows(windowed, _METRICS) if windowed is not None else None
    return RegressorResults(
        **_run_info(name, stream, out, window_size, frame),
        **{key: float(metrics[key]) for key in _METRICS},
    )  # type: ignore[typeddict-item]


def evaluate_regressors(
    stream: Stream,
    learners: Mapping[str, Regressor],
    max_instances: int | None = None,
    window_size: int | None = 1000,
    store_predictions: bool = False,
    store_y: bool = False,
    restart_stream: bool = True,
    progress_bar: bool | tqdm = False,
    batch_size: int = 1,
) -> dict[str, RegressorResults]:
    """Test-then-train many regressors on one pass over a stream.

    The learners get the same instances in turn, so the stream is read once.
    This is useful when reading the stream is slow. It does not use the Java
    loop. The training and testing of the learners is interleaved, so
    ``wallclock`` and ``cpu_time`` are for the whole pass and are the same for
    all the learners. Do not use them to compare the speed of learners.

    >>> from capymoa.regressor import FIMTDD, TargetMean, evaluate_regressors
    >>> from capymoa.datasets import FriedTiny
    >>> stream = FriedTiny()
    >>> learners = {
    ...     "fimtdd": FIMTDD(stream.get_schema()),
    ...     "mean": TargetMean(stream.get_schema()),
    ... }
    >>> results = evaluate_regressors(stream, learners, max_instances=1000)
    >>> list(results)
    ['fimtdd', 'mean']

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
    :param batch_size: Instances per mini-batch, for batch learners.
    :return: The results by name.
    """
    runs = _require_mapping(learners, "evaluate_regressor")
    if restart_stream:
        stream.restart()
    for one in runs.values():
        _check_batch_size(one, batch_size)
    schema = stream.get_schema()
    if not schema.is_regression():
        raise ValueError("The stream is not a regression stream.")

    cumulative = {n: RegressionEvaluator(schema=schema) for n in runs}
    windowed = {
        n: None
        if window_size is None
        else RegressionWindowedEvaluator(schema=schema, window_size=window_size)
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
        batch_size=batch_size,
    )
    return {
        n: _regressor_results(
            n, stream, outs[n], cumulative[n], windowed[n], window_size
        )
        for n in runs
    }


def evaluate_regressor(
    stream: Stream,
    learner: Regressor,
    max_instances: int | None = None,
    window_size: int | None = 1000,
    store_predictions: bool = False,
    store_y: bool = False,
    optimise: bool = True,
    restart_stream: bool = True,
    progress_bar: bool | tqdm = False,
    batch_size: int = 1,
) -> RegressorResults:
    """Test-then-train a regressor on a stream (prequential evaluation).

    Each instance is first used to test the learner, then to train it.

    >>> from capymoa.regressor import FIMTDD, evaluate_regressor
    >>> from capymoa.datasets import FriedTiny
    >>> stream = FriedTiny()
    >>> results = evaluate_regressor(
    ...     stream, FIMTDD(stream.get_schema()), max_instances=1000
    ... )
    >>> print(f"{results['rmse']:.2f}")
    7.36

    To compare learners on one pass over the stream use
    :func:`evaluate_regressors`.

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
    :param optimise: Use the Java loop in MOA if the learner allows it. Needs a
        ``window_size``. The Java loop has no progress bar.
    :param batch_size: Instances per mini-batch, for batch learners.
    :return: The results.
    """
    _require_single(learner, "evaluate_regressors")
    if restart_stream:
        stream.restart()
    schema = stream.get_schema()
    if not schema.is_regression():
        raise ValueError("The stream is not a regression stream.")
    name = str(learner)
    if window_size is not None and _is_fast_mode_compilable(stream, learner, optimise):
        _check_batch_size(learner, batch_size)
        cumulative = RegressionEvaluator(schema=schema)
        windowed = RegressionWindowedEvaluator(schema=schema, window_size=window_size)
        out = _prequential_loop_fast(
            stream,
            learner,
            cumulative,
            windowed,
            max_instances=max_instances,
            window_size=window_size,
            store_y=store_y,
            store_predictions=store_predictions,
        )
        return _regressor_results(name, stream, out, cumulative, windowed, window_size)
    return evaluate_regressors(
        stream,
        {name: learner},
        max_instances=max_instances,
        window_size=window_size,
        store_predictions=store_predictions,
        store_y=store_y,
        restart_stream=False,
        progress_bar=progress_bar,
        batch_size=batch_size,
    )[name]
