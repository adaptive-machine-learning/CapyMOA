"""The test-then-train loop shared by every domain.

Domain ``evaluate_*`` functions build their evaluators, call the loop and turn
its output into a typed result. Nothing here knows about a domain.
"""

import sys
import time
from collections.abc import Callable, Mapping, Sequence, Sized
from dataclasses import dataclass, field
from itertools import islice
from typing import Any, Protocol

import numpy as np
from moa.evaluation import EfficientEvaluationLoops
from moa.streams import InstanceStream
from tqdm import tqdm

from capymoa._utils import batched
from capymoa.base import MOAPredictionIntervalLearner
from capymoa.core import LabeledInstance, RegressionInstance
from capymoa.evaluation._progress_bar import resolve_progress_bar
from capymoa.evaluation.results import RunInfo
from capymoa.stream import Stream
from capymoa.stream.drift import DriftStream, RecurrentConceptDriftStream


class _Evaluator(Protocol):
    """What the loop needs from an evaluator."""

    result_windows: list
    window_size: int | None

    def update(self, y_true: Any, y_pred: Any) -> None: ...
    def get_instances_seen(self) -> int: ...
    def metrics(self) -> list: ...


#: Test and train on a batch of instances. Returns the targets and predictions.
Step = Callable[[Sequence[Any]], tuple[Sequence[Any], Sequence[Any]]]


@dataclass
class _LoopOutput:
    """What the loop measured. The evaluators are updated in place."""

    instances: int
    wallclock: float
    cpu_time: float
    y_true: np.ndarray | None
    y_pred: np.ndarray | None
    #: Extra measurements MOA reports for the run, by name.
    other: dict[str, float] = field(default_factory=dict)


def start_time_measuring() -> tuple[float, float]:
    """Start a wallclock and a CPU timer."""
    return time.time(), time.process_time()


def stop_time_measuring(
    start_wallclock_time: float, start_cpu_time: float
) -> tuple[float, float]:
    """Stop the timers. Returns the elapsed wallclock and CPU time in seconds."""
    return (
        time.time() - start_wallclock_time,
        time.process_time() - start_cpu_time,
    )


def _is_fast_mode_compilable(stream: Stream, learner, optimise=True) -> bool:
    """Check if the stream and learner work with the efficient loops in MOA."""
    # refuse prediction interval learner
    if not hasattr(learner, "moa_learner") or isinstance(
        learner.moa_learner, MOAPredictionIntervalLearner
    ):
        return False

    is_moa_stream = isinstance(stream.get_moa_stream(), InstanceStream)
    is_moa_learner = hasattr(learner, "moa_learner") and learner.moa_learner is not None

    return is_moa_stream and is_moa_learner and optimise


def _get_expected_length(
    stream: Stream, max_instances: int | None = None
) -> int | None:
    """Get the expected length of the stream."""
    if isinstance(stream, Sized) and max_instances is not None:
        return min(len(stream), max_instances)
    elif isinstance(stream, Sized) and max_instances is None:
        return len(stream)
    elif max_instances is not None:
        return max_instances
    else:
        return None


def _setup_progress_bar(
    label: str,
    progress_bar: bool | tqdm,
    stream: Stream,
    max_instances: int | None,
):
    expected_length = _get_expected_length(stream, max_instances)
    progress_bar = resolve_progress_bar(progress_bar, label)
    if progress_bar is not None and expected_length is not None:
        progress_bar.set_total(expected_length)
    return progress_bar


def _drift_info(stream: Stream) -> dict[str, list]:
    """The ``drifts``, ``drift_widths`` and ``concepts`` keys of a stream.

    Empty if the stream has no drifts.
    """
    if not isinstance(stream, DriftStream):
        return {}
    drifts = stream.get_drifts()
    info: dict[str, list] = {
        "drifts": [int(drift.position) for drift in drifts],
        "drift_widths": [int(drift.width or 0) for drift in drifts],
    }
    if isinstance(stream, RecurrentConceptDriftStream):
        info["concepts"] = [
            {"id": str(c["id"]), "start": int(c["start"]), "end": int(c["end"])}
            for c in stream.concept_info
        ]
    return info


def _batch_learner_class(name: str):
    """Return a ``capymoa.base`` batch class, or ``None`` if torch is absent.

    These classes require PyTorch, which is an optional extra. An instance of
    one cannot exist unless its module has already been imported, so checking
    :data:`sys.modules` lets evaluation support batch learners without dragging
    torch into a torch-free install.
    """
    module = sys.modules.get(_BATCH_MODULES[name])
    return getattr(module, name, None) if module is not None else None


_BATCH_MODULES = {
    "Batch": "capymoa.base._batch",
    "BatchClassifier": "capymoa.base._batch_classifier",
    "BatchRegressor": "capymoa.base._batch_regressor",
}


def _isinstance_batch(learner, *names: str) -> bool:
    """``isinstance`` against batch classes, without importing torch."""
    classes = tuple(
        cls for cls in (_batch_learner_class(name) for name in names) if cls is not None
    )
    return bool(classes) and isinstance(learner, classes)


def _get_target(instance: LabeledInstance | RegressionInstance) -> int | np.double:
    """Get the target value from an instance."""
    if isinstance(instance, LabeledInstance):
        return instance.y_index
    elif isinstance(instance, RegressionInstance):
        return instance.y_value
    else:
        raise TypeError("Unknown instance type")


def _to_array(values: list) -> np.ndarray:
    """Make an array. A missing value (``None``) becomes NaN.

    A missing value among sequences (such as prediction intervals) becomes a
    row of NaN.
    """
    try:
        array = np.array(values)
    except ValueError:  # Ragged, such as ``[None, [lo, mid, hi]]``.
        array = np.array(values, dtype=object)
    if array.dtype == object:
        shape = next((np.shape(v) for v in values if v is not None), ())
        missing = np.full(shape, np.nan)
        try:
            return np.array([missing if v is None else v for v in values], dtype=float)
        except (TypeError, ValueError):
            pass
    return array


def _supervised_step(learner) -> Step:
    """Predict, then train on a batch of instances with a supervised learner."""

    def step(batch):
        yb_true = [_get_target(instance) for instance in batch]
        yb_pred = []
        if _isinstance_batch(learner, "Batch"):
            # Collect a batch of instances and predict them all at once
            import torch  # optional extra; a Batch learner guarantees it

            np_x = np.stack([instance.x for instance in batch])
            torch_x = torch.from_numpy(np_x).to(
                device=learner.device, dtype=learner.x_dtype
            )
            torch_y = torch.tensor(
                yb_true, dtype=learner.y_dtype, device=learner.device
            )
            yb_pred = learner.batch_predict(torch_x).tolist()
            learner.batch_train(torch_x, torch_y)
        else:
            for instance in batch:
                yb_pred.append(learner.predict(instance))
                learner.train(instance)
        return yb_true, yb_pred

    return step


def _check_batch_size(learner, batch_size: int) -> None:
    if batch_size != 1 and not _isinstance_batch(
        learner, "BatchClassifier", "BatchRegressor"
    ):
        raise ValueError(
            "The learner is not a batch learner, but batch_size is set to a value greater than 1."
        )


def _prequential_loop(
    stream: Stream,
    steps: Mapping[str, Step],
    cumulative: Mapping[str, _Evaluator],
    windowed: Mapping[str, _Evaluator | None],
    *,
    max_instances: int | None,
    window_size: int | None,
    store_y: bool,
    store_predictions: bool,
    progress_bar: bool | tqdm = False,
    progress_label: str = "Eval",
    batch_size: int = 1,
) -> dict[str, _LoopOutput]:
    """Test-then-train every learner on the stream, going over it once.

    :param steps: How to test and train a learner on a batch, by learner name.
    :param cumulative: The evaluator over the whole stream, by learner name.
    :param windowed: The windowed evaluator, by learner name. ``None`` if
        ``window_size`` is ``None``.
    """
    names = list(steps)
    y_true: dict[str, list] = {n: [] for n in names}
    y_pred: dict[str, list] = {n: [] for n in names}
    instances = 0

    start_wallclock_time, start_cpu_time = start_time_measuring()
    bar = _setup_progress_bar(progress_label, progress_bar, stream, max_instances)
    for batch in batched(islice(stream, max_instances), batch_size):
        for name in names:
            yb_true, yb_pred = steps[name](batch)
            for t, p in zip(yb_true, yb_pred, strict=True):
                cumulative[name].update(t, p)
                if windowed[name] is not None:
                    windowed[name].update(t, p)
            if store_y:
                y_true[name].extend(yb_true)
            if store_predictions:
                y_pred[name].extend(yb_pred)
        instances += len(batch)
        if bar is not None:
            bar.update(len(batch))
    if bar is not None:
        bar.close()
    wallclock, cpu_time = stop_time_measuring(start_wallclock_time, start_cpu_time)

    outputs = {}
    for name in names:
        # Keep the last, shorter window if the window size does not divide the
        # stream.
        win = windowed[name]
        if win is not None and window_size and win.get_instances_seen() % window_size:
            win.result_windows.append(win.metrics())
        outputs[name] = _LoopOutput(
            instances=instances,
            wallclock=wallclock,
            cpu_time=cpu_time,
            y_true=_to_array(y_true[name]) if store_y else None,
            y_pred=_to_array(y_pred[name]) if store_predictions else None,
        )
    return outputs


def _prequential_loop_fast(
    stream: Stream,
    learner,
    cumulative,
    windowed,
    *,
    max_instances: int | None,
    window_size: int,
    store_y: bool,
    store_predictions: bool,
    ssl: tuple[int, int, float, int] | None = None,
) -> _LoopOutput:
    """The test-then-train loop of one learner, run by MOA in Java.

    Needs a MOA learner and a MOA stream (see :func:`_is_fast_mode_compilable`).

    :param ssl: ``(initial_window_size, delay_length, label_probability,
        random_seed)`` for semi-supervised evaluation, else ``None``.
    """
    if not _is_fast_mode_compilable(stream, learner):
        raise ValueError(
            "The fast loop requires the stream object to have a `Stream.moa_stream`"
        )
    start_wallclock_time, start_cpu_time = start_time_measuring()
    limit = -1 if max_instances is None else max_instances
    if ssl is None:
        moa_results = EfficientEvaluationLoops.PrequentialEvaluation(
            stream.moa_stream,
            learner.moa_learner,
            cumulative.moa_basic_evaluator,
            windowed.moa_evaluator,
            limit,
            window_size,
            store_y,
            store_predictions,
        )
    else:
        initial_window_size, delay_length, label_probability, random_seed = ssl
        moa_results = EfficientEvaluationLoops.PrequentialSSLEvaluation(
            stream.moa_stream,
            learner.moa_learner,
            cumulative.moa_basic_evaluator,
            windowed.moa_evaluator,
            limit,
            window_size,
            initial_window_size,
            delay_length,
            label_probability,
            random_seed,
            True,
            store_y,
            store_predictions,
        )
    wallclock, cpu_time = stop_time_measuring(start_wallclock_time, start_cpu_time)

    windowed.result_windows = []
    if moa_results is not None and moa_results.windowedResults is not None:
        for entry in moa_results.windowedResults:
            windowed.result_windows.append(entry)

    metrics = dict(zip(cumulative.metrics_header(), cumulative.metrics()))
    other = moa_results.otherMeasurements or {}
    return _LoopOutput(
        instances=int(metrics["instances"]),
        wallclock=wallclock,
        cpu_time=cpu_time,
        y_true=np.array(moa_results.targets) if store_y else None,
        y_pred=np.array(moa_results.predictions) if store_predictions else None,
        other={str(k): float(v) for k, v in dict(other).items()},
    )


def _require_mapping(learners, singular: str) -> dict[str, Any]:
    """Check ``learners`` is a mapping of names to learners."""
    if not isinstance(learners, Mapping):
        raise TypeError(
            f"`learners` must map names to learners, got {type(learners).__name__}. "
            f"Use `{singular}` for a single learner."
        )
    if not learners:
        raise ValueError("No learners to evaluate.")
    return dict(learners)


def _require_single(learner, plural: str) -> None:
    """Check ``learner`` is one learner and not a mapping of them."""
    if isinstance(learner, Mapping):
        raise TypeError(f"Got a mapping of learners. Use `{plural}` for many learners.")


def _progress_label(prefix: str, learners: Mapping[str, Any], stream: Stream) -> str:
    stream_name = type(stream).__name__
    if len(learners) == 1:
        (learner,) = learners.values()
        return f"{prefix} {type(learner).__name__!r} on {stream_name!r}"
    return f"{prefix} {len(learners)} learners on {stream_name}"


def _windows(evaluator, columns: Sequence[str]) -> dict[str, np.ndarray]:
    """The windows of an evaluator in columns: ``instances`` and the given metrics."""
    frame = evaluator.metrics_per_window()
    windows = {"instances": frame["instances"].to_numpy().astype(int)}
    for column in columns:
        windows[column] = frame[column].to_numpy(dtype=float)
    return windows


def _run_info(
    name: str,
    stream: "Stream | str",
    out: _LoopOutput,
    window_size: int | None,
    windowed: Mapping[str, np.ndarray] | None,
) -> RunInfo:
    """Run info of a stream, or of a stream known only by its name.

    Optional keys are left out when there is nothing to put in them.
    """
    info = RunInfo(
        learner=name,
        stream=stream if isinstance(stream, str) else str(stream),
        instances=out.instances,
        wallclock=out.wallclock,
        cpu_time=out.cpu_time,
    )  # type: ignore[typeddict-item]
    optional = {
        "window_size": window_size,
        "windowed": windowed,
        "y_true": out.y_true,
        "y_pred": out.y_pred,
    }
    info.update({k: v for k, v in optional.items() if v is not None})  # type: ignore[typeddict-item]
    if not isinstance(stream, str):
        info.update(_drift_info(stream))  # type: ignore[typeddict-item]
    return info
