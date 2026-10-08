"""The test-then-train loop shared by every domain.

Domain ``evaluate_*`` functions build their evaluators, call the loop and turn
its output into a typed result. Nothing here knows about a domain.
"""

import sys
import time
from collections.abc import Mapping, Sequence, Sized
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


@dataclass
class _LoopOutput:
    """What the loop measured. The evaluators are updated in place."""

    instances: int
    wallclock: float
    cpu_time: float
    y_true: list | None
    y_pred: list | None
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
    moa_learner = getattr(learner, "moa_learner", None)
    return (
        optimise
        and moa_learner is not None
        # refuse prediction interval learner
        and not isinstance(moa_learner, MOAPredictionIntervalLearner)
        and isinstance(stream.get_moa_stream(), InstanceStream)
    )


def _get_expected_length(
    stream: Stream, max_instances: int | None = None
) -> int | None:
    """Get the expected length of the stream."""
    if isinstance(stream, Sized):
        return len(stream) if max_instances is None else min(len(stream), max_instances)
    return max_instances


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


def _is_batch(learner) -> bool:
    """``isinstance(learner, Batch)`` without importing torch.

    :class:`~capymoa.base.Batch` needs PyTorch, an optional extra. A ``Batch``
    learner cannot exist unless its module is already imported, so checking
    :data:`sys.modules` avoids importing torch.
    """
    module = sys.modules.get("capymoa.base._batch")
    return module is not None and isinstance(learner, module.Batch)


def _get_target(instance: LabeledInstance | RegressionInstance) -> int | np.double:
    """Get the target value from an instance."""
    if isinstance(instance, LabeledInstance):
        return instance.y_index
    elif isinstance(instance, RegressionInstance):
        return instance.y_value
    else:
        raise TypeError("Unknown instance type")


class _Run:
    """One learner in the test-then-train loop, with its evaluators.

    Override :meth:`test_then_train` to change how the learner is tested and
    trained.
    """

    def __init__(
        self,
        learner,
        cumulative: _Evaluator,
        windowed: _Evaluator | None,
        *,
        store_y: bool,
        store_predictions: bool,
    ):
        self.learner = learner
        self.cumulative = cumulative
        self.windowed = windowed
        self.store_y = store_y
        self.store_predictions = store_predictions
        self._y_true: list = []
        self._y_pred: list = []

    def test_then_train(
        self, batch: Sequence[Any]
    ) -> tuple[Sequence[Any], Sequence[Any]]:
        """Test, then train on a batch. Returns the targets and predictions."""
        learner = self.learner
        yb_true = [_get_target(instance) for instance in batch]
        yb_pred = []
        if _is_batch(learner):
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

    def step(self, batch: Sequence[Any]) -> None:
        """Test-then-train on a batch, then update the evaluators."""
        y_true, y_pred = self.test_then_train(batch)
        for t, p in zip(y_true, y_pred, strict=True):
            self.cumulative.update(t, p)
            if self.windowed is not None:
                self.windowed.update(t, p)
        if self.store_y:
            self._y_true.extend(y_true)
        if self.store_predictions:
            self._y_pred.extend(y_pred)

    def output(self, instances: int, wallclock: float, cpu_time: float) -> _LoopOutput:
        """What the loop measured for this learner."""
        # Keep the last, shorter window if the window size does not divide the
        # stream.
        win = self.windowed
        if (
            win is not None
            and win.window_size
            and win.get_instances_seen() % win.window_size
        ):
            win.result_windows.append(win.metrics())
        return _LoopOutput(
            instances=instances,
            wallclock=wallclock,
            cpu_time=cpu_time,
            y_true=self._y_true if self.store_y else None,
            y_pred=self._y_pred if self.store_predictions else None,
        )


def _check_batch_size(learner, batch_size: int) -> None:
    if batch_size != 1 and not _is_batch(learner):
        raise ValueError(
            "The learner is not a batch learner, but batch_size is set to a value greater than 1."
        )


def _prequential_loop(
    stream: Stream,
    runs: Mapping[str, _Run],
    *,
    max_instances: int | None,
    progress_bar: bool | tqdm = False,
    progress_label: str = "Eval",
    batch_size: int = 1,
) -> dict[str, _LoopOutput]:
    """Test-then-train every learner on the stream, going over it once.

    This only drives the stream. Each :class:`_Run` tests, trains and keeps
    its own evaluators.

    :param runs: The learners to run, by name.
    """
    instances = 0

    start_wallclock_time, start_cpu_time = start_time_measuring()
    bar = _setup_progress_bar(progress_label, progress_bar, stream, max_instances)
    for batch in batched(islice(stream, max_instances), batch_size):
        for run in runs.values():
            run.step(batch)
        instances += len(batch)
        if bar is not None:
            bar.update(len(batch))
    if bar is not None:
        bar.close()
    wallclock, cpu_time = stop_time_measuring(start_wallclock_time, start_cpu_time)

    return {
        name: run.output(instances, wallclock, cpu_time) for name, run in runs.items()
    }


def _prequential_loop_fast(
    stream: Stream,
    run: _Run,
    *,
    max_instances: int | None,
    ssl: tuple[int, int, float, int] | None = None,
) -> _LoopOutput:
    """The test-then-train loop of one learner, run by MOA in Java.

    Needs a MOA learner and a MOA stream (see :func:`_is_fast_mode_compilable`).
    The run needs a windowed evaluator. It must not override
    :meth:`_Run.test_then_train`, since Java cannot run that code.

    :param ssl: ``(initial_window_size, delay_length, label_probability,
        random_seed)`` for semi-supervised evaluation, else ``None``.
    """
    if type(run).test_then_train is not _Run.test_then_train:
        raise TypeError("The Java loop cannot run a custom test_then_train.")
    learner, cumulative, windowed = run.learner, run.cumulative, run.windowed
    if windowed is None:
        raise ValueError("The fast loop requires a windowed evaluator.")
    if not _is_fast_mode_compilable(stream, learner):
        raise ValueError(
            "The fast loop requires the stream object to have a `Stream.moa_stream`"
        )
    window_size = windowed.window_size
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
            run.store_y,
            run.store_predictions,
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
            run.store_y,
            run.store_predictions,
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
        # Via numpy to turn Java arrays into Python numbers of the same type.
        y_true=np.array(moa_results.targets).tolist() if run.store_y else None,
        y_pred=(
            np.array(moa_results.predictions).tolist()
            if run.store_predictions
            else None
        ),
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
    windowed: _Evaluator | None,
    columns: Sequence[str],
) -> RunInfo:
    """Run info of a stream, or of a stream known only by its name.

    Optional keys are left out when there is nothing to put in them.

    :param windowed: The windowed evaluator, or ``None`` if windows are off.
    :param columns: The metric columns to put in ``windowed``.
    """
    window_size = windowed.window_size if windowed is not None else None
    frame = _windows(windowed, columns) if windowed is not None else None
    info = RunInfo(
        learner=name,
        stream=stream if isinstance(stream, str) else str(stream),
        instances=out.instances,
        wallclock=out.wallclock,
        cpu_time=out.cpu_time,
    )  # type: ignore[typeddict-item]
    optional = {
        "window_size": window_size,
        "windowed": frame,
        "y_true": out.y_true,
        "y_pred": out.y_pred,
    }
    info.update({k: v for k, v in optional.items() if v is not None})  # type: ignore[typeddict-item]
    if not isinstance(stream, str):
        info.update(_drift_info(stream))  # type: ignore[typeddict-item]
    return info
