"""Main OCL train/evaluation loop (event-driven rewrite)."""

from typing import Optional, Sequence, Tuple

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from capymoa.base import BatchClassifier, Classifier
from capymoa.instance import Instance, LabeledInstance
from capymoa.ocl.events import Event, Handler, Dispatcher
from capymoa.type_alias import LabelIndex

from ._metrics_handler import _OCLMetricsHandler
from ._metrics import OCLMetrics
from . import events


def _abstain_prediction_uniform(rng: np.random.Generator, n_classes: int) -> LabelIndex:
    return int(rng.integers(0, n_classes))


def _batch_test(rng: np.random.Generator, learner: Classifier, x: Tensor) -> np.ndarray:
    """Test a batch of instances using the learner."""
    batch_size = x.shape[0]
    x = x.view(batch_size, -1)
    if isinstance(learner, BatchClassifier):
        x = x.to(dtype=learner.x_dtype, device=learner.device)
        return learner.batch_predict(x).cpu().numpy()
    else:
        yb_pred = np.zeros(batch_size, dtype=int)
        for i in range(batch_size):
            instance = Instance.from_array(learner.schema, x[i].numpy())
            y_pred = learner.predict(instance)
            if y_pred is None:
                y_pred = _abstain_prediction_uniform(
                    rng, learner.schema.get_num_classes()
                )
            yb_pred[i] = y_pred
        return yb_pred


def _batch_train(learner: Classifier, x: Tensor, y: Tensor):
    """Train a batch of instances using the learner."""
    batch_size = x.shape[0]
    x = x.view(batch_size, -1)
    if isinstance(learner, BatchClassifier):
        x = x.to(dtype=learner.x_dtype, device=learner.device)
        y = y.to(dtype=learner.y_dtype, device=learner.device)
        learner.batch_train(x, y)
    else:
        for i in range(batch_size):
            instance = LabeledInstance.from_array(
                learner.schema, x[i].numpy(), int(y[i].item())
            )
            learner.train(instance)


class _ProgressBarSink(Handler):
    """Event sink that manages progress bar updates and lifecycle."""

    def __init__(self, total: int):
        self._pbar = tqdm(total=total, desc="Train & Eval")

    @staticmethod
    def from_streams(
        train_streams: Sequence[DataLoader[Tuple[Tensor, Tensor]]],
        test_streams: Sequence[DataLoader[Tuple[Tensor, Tensor]]],
        epochs: int,
        continual_evaluations: int,
    ) -> "_ProgressBarSink":
        train_len = sum(len(s) for s in train_streams)
        test_len = sum(len(s) for s in test_streams)
        total = train_len * epochs + test_len * continual_evaluations * len(
            test_streams
        )
        return _ProgressBarSink(total=total)

    def attach_with(self, source: Dispatcher) -> "_ProgressBarSink":
        source.subscribe(events.TrainBatchPredict, self._on_batch)
        source.subscribe(events.EvalBatchPredict, self._on_batch)
        source.subscribe(events.TrainEnd, self._on_loop_end)
        return self

    def _on_batch(self, _: Event) -> None:
        self._pbar.update(1)

    def _on_loop_end(self, _: Event) -> None:
        self._pbar.close()


def ocl_train_eval_loop(
    learner: Classifier,
    train_streams: Sequence[DataLoader[Tuple[Tensor, Tensor]]],
    test_streams: Sequence[DataLoader[Tuple[Tensor, Tensor]]],
    continual_evaluations: int = 1,
    progress_bar: bool = False,
    eval_window_size: int = 1000,
    epochs: int = 1,
    dispatcher: Optional[Dispatcher] = None,
) -> OCLMetrics:
    """Run the OCL training loop with periodic continual evaluation.

    :param learner: The classifier to train and evaluate.
    :param train_streams: Sequence of task-wise training data loaders.
    :param test_streams: Sequence of task-wise test data loaders. Must have the same
        number of tasks as `train_streams`.
    :param continual_evaluations: Number of evaluation passes performed during each
        training task, defaults to 1.
    :param progress_bar: Whether to enable a progress-bar, defaults to False.
    :param eval_window_size: Window size used by the default metrics handler to compute
        rolling metrics, defaults to 1000.
    :param epochs: Number of epochs to train each task stream, defaults to 1.
    :param dispatcher: Optional event dispatcher. If None, a new dispatcher is created.
    :return: Aggregated OCL metrics collected by the default metrics handler.
    :raises ValueError: If train/test task counts differ, learner is not a classifier,
        ``continual_evaluations < 1``, or a train stream has fewer batches than
        requested evaluations.
    """
    epochs = epochs or 1
    n_tasks = len(train_streams)
    if n_tasks != len(test_streams):
        raise ValueError("Number of train and test tasks must be equal")
    if not isinstance(learner, Classifier):
        raise ValueError("Learner must be a classifier")
    if 1 > continual_evaluations:
        raise ValueError("Continual evaluations must be at least 1")
    if (min_stream_len := min(len(s) for s in train_streams)) < continual_evaluations:
        raise ValueError(
            "Cannot evaluate more times than the number of batches. "
            f"(min stream length (in batches): {min_stream_len}, "
            f"continual evaluations: {continual_evaluations})"
        )

    dispatcher = dispatcher or Dispatcher()
    assert isinstance(dispatcher, Dispatcher)

    default_sink = _OCLMetricsHandler(
        learner=learner,
        task_count=n_tasks,
        continual_evaluations=continual_evaluations,
        eval_window_size=eval_window_size,
    )
    default_sink.attach_with(dispatcher)
    if isinstance(learner, Handler):
        learner.attach_with(dispatcher)

    if progress_bar:
        _ProgressBarSink.from_streams(
            train_streams, test_streams, epochs, continual_evaluations
        ).attach_with(dispatcher)

    rng = np.random.default_rng(learner.random_seed)
    dispatcher.notify(events.TrainBegin())

    global_step = 0

    def evaluate_learner(eval_step: int, global_step: int) -> None:
        dispatcher.notify(events.TestBegin())
        for test_task_id, test_stream in enumerate(test_streams):
            dispatcher.notify(
                events.TestTaskBegin(
                    train_task_id, test_task_id, eval_step, global_step
                )
            )

            for batch_id, (x_test, y_test) in enumerate(test_stream):
                y_hat = torch.from_numpy(_batch_test(rng, learner, x_test))
                dispatcher.notify(
                    events.EvalBatchPredict(
                        train_task=train_task_id,
                        test_task=test_task_id,
                        continual_eval=eval_step,
                        global_step=global_step,
                        batch=batch_id,
                        x=x_test,
                        y=y_test,
                        y_hat=y_hat,
                    )
                )

            dispatcher.notify(
                events.TestTaskEnd(train_task_id, test_task_id, eval_step, global_step)
            )
        dispatcher.notify(events.TestEnd())

    for train_task_id, train_stream in enumerate(train_streams):
        dispatcher.notify(events.TrainTaskBegin(train_task_id, global_step))

        eval_interval = (len(train_stream) * epochs) // continual_evaluations
        step = 0

        for _ in range(epochs):
            for batch_id, (x_train, y_train) in enumerate(train_stream):
                y_hat = torch.from_numpy(_batch_test(rng, learner, x_train))
                dispatcher.notify(
                    events.TrainBatchPredict(
                        train_task=train_task_id,
                        global_step=global_step,
                        batch=batch_id,
                        x=x_train,
                        y=y_train,
                        y_hat=y_hat,
                    )
                )
                _batch_train(learner, x_train, y_train)

                if (step + 1) % eval_interval == 0:
                    eval_step = step // eval_interval
                    if eval_step < continual_evaluations:
                        evaluate_learner(eval_step=eval_step, global_step=global_step)
                step += 1
                global_step += 1

        dispatcher.notify(events.TrainTaskEnd(train_task_id, global_step))

    dispatcher.notify(events.TrainEnd())

    return default_sink.build(
        learner_name=str(learner),
        stream_name=f"{train_streams[0]}x{len(train_streams)}",
    )
