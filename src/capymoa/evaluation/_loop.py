"""The test-then-train loop shared by every domain.

Domain ``evaluate_*`` functions build their evaluators, call the loop and turn
its output into a typed result. Nothing here knows about a domain.
"""

import math
import sys
import time
from collections.abc import Mapping, Sequence, Sized
from dataclasses import dataclass, field
from itertools import islice
from typing import Any

import numpy as np
from moa.evaluation import EfficientEvaluationLoops, HookableEvaluationLoop
from moa.streams import InstanceStream
from tqdm import tqdm

from capymoa._utils import batched
from capymoa.base import Classifier, MOAPredictionIntervalLearner
from capymoa.core import LabeledInstance
from capymoa.evaluation._abstain import batch_proba_abstains, proba_abstains
from capymoa.evaluation.results import RunInfo
from capymoa.stream import Stream
from capymoa.stream.drift import DriftStream, RecurrentConceptDriftStream


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


def _use_java_loop(
    stream: Stream,
    learner,
    *,
    optimise: bool,
    window_size: int | None,
    batch_size: int = 1,
) -> bool:
    """Whether MOA can run the whole loop in Java."""
    moa_learner = getattr(learner, "moa_learner", None)
    return (
        optimise
        and window_size is not None
        and batch_size == 1
        and moa_learner is not None
        # refuse prediction interval learner
        and not isinstance(moa_learner, MOAPredictionIntervalLearner)
        and isinstance(stream.get_moa_stream(), InstanceStream)
    )


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


class _Run:
    """One learner in the test-then-train loop, with its evaluators.

    Override :meth:`test_then_train` to change how the learner is tested and
    trained.
    """

    def __init__(
        self,
        learner,
        cumulative,
        windowed,
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
        #: Whether each prediction of the last batch was an abstention. Empty when
        #: the learner does not report it.
        self._abstained: list[bool] = []

    def predict(self, instance) -> Any:
        """Predict one instance and note if the classifier abstained.

        A classifier abstains when its confidence is equal to or below chance.
        The note goes in ``self._abstained``.
        """
        learner = self.learner
        if not isinstance(learner, Classifier):
            return learner.predict(instance)
        try:
            proba = learner.predict_proba(instance)
        except NotImplementedError:  # A pipeline has no probabilities.
            return learner.predict(instance)
        schema = getattr(learner, "schema", None)  # OSNN has none
        n = np.size(proba) if schema is None else schema.get_num_classes()
        self._abstained.append(proba_abstains(proba, n))
        if type(learner).predict is Classifier.predict:
            return int(np.argmax(proba))  # What predict would do, without a 2nd call.
        return learner.predict(instance)

    def test_then_train(
        self, batch: Sequence[Any]
    ) -> tuple[Sequence[Any], Sequence[Any]]:
        """Test, then train on a batch. Returns the targets and predictions."""
        learner = self.learner
        yb_true = [
            i.y_index if isinstance(i, LabeledInstance) else i.y_value for i in batch
        ]
        yb_pred = []
        if _is_batch(learner):
            # Collect a batch of instances and predict them all at once
            import torch  # optional extra; a Batch learner guarantees it

            from capymoa.base import BatchClassifier

            np_x = np.stack([instance.x for instance in batch])
            torch_x = torch.from_numpy(np_x).to(
                device=learner.device, dtype=learner.x_dtype
            )
            torch_y = torch.tensor(
                yb_true, dtype=learner.y_dtype, device=learner.device
            )
            if type(learner).batch_predict is BatchClassifier.batch_predict:
                proba = learner.batch_predict_proba(torch_x)
                yb_pred = proba.argmax(1).tolist()
                self._abstained.extend(
                    batch_proba_abstains(proba.detach().cpu().numpy()).tolist()
                )
            else:
                yb_pred = learner.batch_predict(torch_x).tolist()
            learner.batch_train(torch_x, torch_y)
        else:
            for instance in batch:
                yb_pred.append(self.predict(instance))
                learner.train(instance)
        return yb_true, yb_pred

    def warm(self, batch: Sequence[Any]) -> None:
        """Train on a batch. Nothing is tested or measured."""
        learner = self.learner
        if _is_batch(learner):
            import torch  # optional extra; a Batch learner guarantees it

            np_x = np.stack([instance.x for instance in batch])
            torch_x = torch.from_numpy(np_x).to(
                device=learner.device, dtype=learner.x_dtype
            )
            y = [
                i.y_index if isinstance(i, LabeledInstance) else i.y_value
                for i in batch
            ]
            torch_y = torch.tensor(y, dtype=learner.y_dtype, device=learner.device)
            learner.batch_train(torch_x, torch_y)
        else:
            for instance in batch:
                learner.train(instance)

    def _scored(self, y_pred: Sequence[Any]) -> Sequence[Any]:
        """The predictions as scored. A regressor's ``NaN`` is its ``nan_value``."""
        nan_value = getattr(self.cumulative, "nan_value", None)
        if nan_value is None:
            return y_pred
        return [nan_value if math.isnan(p) else p for p in y_pred]

    def step(self, batch: Sequence[Any]) -> None:
        """Test-then-train on a batch, then update the evaluators."""
        self._abstained = []
        y_true, y_pred = self.test_then_train(batch)
        # Only classifiers report abstentions. A regressor's are its NaN.
        extra = [{"abstained": a} for a in self._abstained] or [{}] * len(y_true)
        for t, p, kwargs in zip(y_true, y_pred, extra, strict=True):
            self.cumulative.update(t, p, **kwargs)
            if self.windowed is not None:
                self.windowed.update(t, p, **kwargs)
        if self.store_y:
            self._y_true.extend(y_true)
        if self.store_predictions:
            self._y_pred.extend(self._scored(y_pred))

    def output(self, instances: int, wallclock: float, cpu_time: float) -> _LoopOutput:
        """What the loop measured for this learner."""
        # Keep the last, shorter window if the window size does not divide the
        # stream.
        win = self.windowed
        if win is not None and win.get_instances_seen() % win.window_size:
            win.result_windows.append(win.metrics())
        return _LoopOutput(
            instances=instances,
            wallclock=wallclock,
            cpu_time=cpu_time,
            y_true=self._y_true if self.store_y else None,
            y_pred=self._y_pred if self.store_predictions else None,
        )


def _progress_bar(
    progress_bar: bool | tqdm,
    prefix: str,
    runs: Mapping[str, _Run],
    stream: Stream,
    max_instances: int | None,
) -> tqdm:
    """The progress bar of a loop. It prints nothing if ``progress_bar`` is false."""
    if isinstance(progress_bar, tqdm):
        bar = progress_bar
    else:
        stream_name = type(stream).__name__
        if len(runs) == 1:
            (run,) = runs.values()
            label = f"{prefix} {type(run.learner).__name__!r} on {stream_name!r}"
        else:
            label = f"{prefix} {len(runs)} learners on {stream_name}"
        bar = tqdm(desc=label, disable=not progress_bar)
    total = max_instances
    if isinstance(stream, Sized):
        total = (
            len(stream) if max_instances is None else min(len(stream), max_instances)
        )
    if total is not None:
        bar.total = total
    return bar


def _prequential_loop(
    stream: Stream,
    runs: Mapping[str, _Run],
    *,
    max_instances: int | None,
    progress_bar: bool | tqdm = False,
    progress_prefix: str = "Eval",
    batch_size: int = 1,
    warmup: int = 0,
) -> dict[str, _LoopOutput]:
    """Test-then-train every learner on the stream, going over it once.

    This only drives the stream. Each :class:`_Run` tests, trains and keeps
    its own evaluators.

    :param runs: The learners to run, by name.
    :param warmup: Number of instances, from the start, to train on without
        testing. They count towards ``max_instances``.
    """
    if warmup < 0:
        raise ValueError("warmup must be zero or positive.")
    if batch_size != 1 and not all(_is_batch(run.learner) for run in runs.values()):
        raise ValueError(
            "The learner is not a batch learner, but batch_size is set to a value greater than 1."
        )
    instances = 0
    seen = 0
    start_wallclock, start_cpu = time.time(), time.process_time()
    bar = _progress_bar(progress_bar, progress_prefix, runs, stream, max_instances)
    for batch in batched(islice(stream, max_instances), batch_size):
        n_warm = min(max(warmup - seen, 0), len(batch))
        seen += len(batch)
        if n_warm:
            for run in runs.values():
                run.warm(batch[:n_warm])
            batch = batch[n_warm:]
        if batch:
            for run in runs.values():
                run.step(batch)
        instances += len(batch)
        bar.update(n_warm + len(batch))
    bar.close()
    wallclock = time.time() - start_wallclock
    cpu_time = time.process_time() - start_cpu

    return {
        name: run.output(instances, wallclock, cpu_time) for name, run in runs.items()
    }


def _prequential_loop_fast(
    stream: Stream,
    run: _Run,
    *,
    max_instances: int | None,
    ssl: tuple[int, int, float, int] | None = None,
    warmup: int = 0,
    nan_value: float | None = 0.0,
) -> _LoopOutput:
    """The test-then-train loop of one learner, run by MOA in Java.

    Callers must check :func:`_use_java_loop` first. The run needs a windowed
    evaluator. It must not override :meth:`_Run.test_then_train`, since Java
    cannot run that code.

    :param ssl: ``(initial_window_size, delay_length, label_probability,
        random_seed)`` for semi-supervised evaluation, else ``None``.
    :param warmup: Number of instances, from the start, to train on without
        testing. They count towards ``max_instances``. Not for ``ssl``.
    :param nan_value: For regression, the value that replaces a ``NaN``
        prediction (an abstention). ``None`` keeps the ``NaN``.
    """
    if warmup < 0:
        raise ValueError("warmup must be zero or positive.")
    if warmup and ssl is not None:
        raise ValueError("warmup is not supported for semi-supervised evaluation.")
    if type(run).test_then_train is not _Run.test_then_train:
        raise TypeError("The Java loop cannot run a custom test_then_train.")
    learner, cumulative, windowed = run.learner, run.cumulative, run.windowed
    window_size = windowed.window_size
    start_wallclock, start_cpu = time.time(), time.process_time()
    limit = -1 if max_instances is None else max_instances
    if ssl is None:
        # Only these take a warmup. ``PrequentialEvaluation`` picks one of them
        # by the type of the target, as done here.
        if stream.get_schema().is_classification():
            moa_results = EfficientEvaluationLoops.prequentialEvaluationClassification(
                stream.moa_stream,
                learner.moa_learner,
                cumulative.moa_basic_evaluator,
                windowed.moa_evaluator,
                limit,
                window_size,
                run.store_y,
                run.store_predictions,
                warmup,
                HookableEvaluationLoop(),
            )
        else:
            moa_results = EfficientEvaluationLoops.prequentialEvaluationRegression(
                stream.moa_stream,
                learner.moa_learner,
                cumulative.moa_basic_evaluator,
                windowed.moa_evaluator,
                limit,
                window_size,
                run.store_y,
                run.store_predictions,
                warmup,
                float("nan") if nan_value is None else float(nan_value),
                HookableEvaluationLoop(),
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
    wallclock = time.time() - start_wallclock
    cpu_time = time.process_time() - start_cpu

    windowed.result_windows = list(moa_results.windowedResults or [])
    other = moa_results.otherMeasurements or {}
    return _LoopOutput(
        instances=int(cumulative.metrics_dict()["instances"]),
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


def _results_body(
    name: str,
    stream: "Stream | str",
    out: _LoopOutput,
    cumulative,
    windowed,
    windows: type,
    whole_stream_only: Sequence[str] = (),
) -> dict[str, Any]:
    """Run info, then the whole stream metrics under ``cumulative``.

    The stream may be a name only. Optional keys are left out when there is
    nothing to put in them.

    :param cumulative: The evaluator over the whole stream.
    :param windowed: The windowed evaluator, or ``None`` if windows are off.
    :param windows: The ``*Windows`` TypedDict. Its keys, except ``instances``,
        are the metrics to take from both evaluators.
    :param whole_stream_only: Metrics that only the whole stream evaluator
        reports, so they are not in ``windows``.
    """
    # The metrics are the array keys, except ``instances``. Nested keys, such as
    # ``cumulative_per_class``, are built by the domain.
    metrics = [
        key
        for key, kind in windows.__annotations__.items()
        if kind is np.ndarray and key != "instances"
    ]
    info = RunInfo(
        learner=name,
        stream=stream if isinstance(stream, str) else str(stream),
        instances=out.instances,
        wallclock=out.wallclock,
        cpu_time=out.cpu_time,
    )  # type: ignore[typeddict-item]
    if windowed is not None:
        header = windowed.metrics_header()
        rows = windowed.result_windows
        table = np.asarray(rows, dtype=float).reshape(len(rows), len(header))
        column = {name: table[:, i] for i, name in enumerate(header)}
        # The window evaluators count the whole stream in ``abstentions``, and
        # the current window in ``abstentions_window``.
        if "abstentions_window" in column:
            column["abstentions"] = column["abstentions_window"]
        info["window_size"] = windowed.window_size
        info["windowed"] = {
            "instances": column["instances"].astype(int),
            **{key: column[key] for key in metrics if key in column},
        }
    if out.y_true is not None:
        info["y_true"] = out.y_true
    if out.y_pred is not None:
        info["y_pred"] = out.y_pred
    if not isinstance(stream, str):
        info.update(_drift_info(stream))  # type: ignore[typeddict-item]
    # A metric that only the windowed evaluator reports is not in the whole
    # stream values, so it is left out of ``cumulative``.
    values = cumulative.metrics_dict()
    keys = [*metrics, *whole_stream_only]
    return {
        **info,
        "cumulative": {
            key: int(values[key]) if key == "abstentions" else float(values[key])
            for key in keys
            if key in values
        },
    }
