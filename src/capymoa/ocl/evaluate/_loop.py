"""Main OCL train/evaluation loop (event-driven rewrite)."""

from collections.abc import Sequence

import numpy as np
from torch import Tensor
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from capymoa.base import BatchClassifier, Classifier
from capymoa.core import Instance, LabeledInstance
from capymoa.ocl.events import Dispatcher, Event, Handler

from . import events


def _batch_test(learner: Classifier, x: Tensor) -> np.ndarray:
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
            yb_pred[i] = learner.predict(instance)
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
        train_streams: Sequence[DataLoader[tuple[Tensor, Tensor]]],
        test_streams: Sequence[DataLoader[tuple[Tensor, Tensor]]],
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
