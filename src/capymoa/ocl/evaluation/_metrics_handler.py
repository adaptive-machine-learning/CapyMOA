"""Default event sink for OCL metrics collection."""

from typing import Optional

import torch

from capymoa.base import Classifier
from capymoa.ocl.events import Event, Handler, Dispatcher
from capymoa.evaluation.evaluation import (
    ClassificationEvaluator,
    ClassificationWindowedEvaluator,
    start_time_measuring,
    stop_time_measuring,
)
from capymoa.evaluation.results import PrequentialResults
from capymoa.ocl.evaluation.events import (
    TrainEnd,
    TrainBegin,
    TrainBatchPredict,
    TrainTaskEnd,
)

from ._evaluator import _OCLEvaluator
from ._metrics import OCLMetrics


class _OCLMetricsHandler(Handler):
    """Event sink that reproduces the legacy OCL metric collection."""

    def __init__(
        self,
        learner: Classifier,
        task_count: int,
        continual_evaluations: int,
        eval_window_size: int,
    ):
        self._collector = _OCLEvaluator(
            task_count,
            continual_evaluations,
            learner.schema.get_num_classes(),
        )
        self._online_eval = ClassificationEvaluator(schema=learner.schema)
        self._windowed_eval = ClassificationWindowedEvaluator(
            schema=learner.schema, window_size=eval_window_size
        )
        self._boundary_instances = torch.zeros(task_count + 1)
        self._start_wallclock_time: Optional[float] = None
        self._start_cpu_time: Optional[float] = None
        self._elapsed_wallclock_time: float = 0.0
        self._elapsed_cpu_time: float = 0.0

    def attach_with(self, source: Dispatcher) -> "_OCLMetricsHandler":
        self._collector.attach_with(source)
        source.subscribe(TrainBegin, self._on_loop_start)
        source.subscribe(TrainEnd, self._on_loop_end)
        source.subscribe(TrainBatchPredict, self._on_train_batch)
        source.subscribe(TrainTaskEnd, self._on_train_task_end)
        return self

    def _on_loop_start(self, _: Event) -> None:
        self._start_wallclock_time, self._start_cpu_time = start_time_measuring()

    def _on_loop_end(self, _: Event) -> None:
        if self._start_wallclock_time is None or self._start_cpu_time is None:
            return
        self._elapsed_wallclock_time, self._elapsed_cpu_time = stop_time_measuring(
            self._start_wallclock_time, self._start_cpu_time
        )

    def _on_train_batch(self, event: TrainBatchPredict) -> None:
        for y_true, y_pred in zip(event.y, event.y_hat, strict=True):
            y_true_i = int(y_true.item())
            y_pred_i = int(y_pred.item())
            self._online_eval.update(y_true_i, y_pred_i)
            self._windowed_eval.update(y_true_i, y_pred_i)

    def _on_train_task_end(self, event: TrainTaskEnd) -> None:
        self._boundary_instances[event.train_task + 1] = self.instances_seen

    @property
    def instances_seen(self) -> int:
        return self._online_eval.instances_seen

    def build(self, learner_name: str, stream_name: str) -> OCLMetrics:
        return self._collector.build(
            PrequentialResults(
                learner=learner_name,
                stream=stream_name,  # type: ignore[arg-type]
                cumulative_evaluator=self._online_eval,
                windowed_evaluator=self._windowed_eval,
                wallclock=self._elapsed_wallclock_time,
                cpu_time=self._elapsed_cpu_time,
            ),
            self._boundary_instances,
        )
