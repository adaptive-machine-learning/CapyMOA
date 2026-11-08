"""Evaluate an online continual learner on task-wise train and test streams."""

from collections.abc import Sequence

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import DataLoader

from capymoa.base import Classifier
from capymoa.ocl.evaluate import events
from capymoa.ocl.evaluate._loop import _batch_test, _batch_train, _ProgressBarSink
from capymoa.ocl.evaluate._metrics_handler import _OCLMetricsHandler
from capymoa.ocl.evaluate._results import OCLResults
from capymoa.ocl.events import Dispatcher, Handler


def evaluate_ocl(
    learner: Classifier,
    train_streams: Sequence[DataLoader[tuple[Tensor, Tensor]]],
    test_streams: Sequence[DataLoader[tuple[Tensor, Tensor]]],
    continual_evaluations: int = 1,
    progress_bar: bool = False,
    eval_window_size: int = 1000,
    epochs: int = 1,
    dispatcher: Dispatcher | None = None,
) -> OCLResults:
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
    :return: The :class:`~capymoa.ocl.evaluate.OCLResults` collected by the default metrics handler.
    :raises ValueError: If train/test task counts differ, ``continual_evaluations
        < 1``, or a train stream has fewer batches than requested evaluations.
    :raises TypeError: If learner is not a classifier.
    """
    epochs = epochs or 1
    n_tasks = len(train_streams)
    if n_tasks != len(test_streams):
        raise ValueError("Number of train and test tasks must be equal")
    if not isinstance(learner, Classifier):
        raise TypeError("Learner must be a classifier")
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
