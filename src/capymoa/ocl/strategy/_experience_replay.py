from typing import Tuple

import numpy as np
import torch
from torch import Tensor

from capymoa.base import BatchClassifier
from capymoa.classifier import Finetune
from capymoa.ocl.base import TaskAware, TaskBoundaryAware


class _ReservoirSampler:
    def __init__(self, item_count: int, feature_count: int, rng: torch.Generator):
        self.item_count = item_count
        self.feature_count = feature_count
        self.reservoir_x = torch.zeros((item_count, feature_count))
        self.reservoir_y = torch.zeros((item_count,), dtype=torch.long)
        self.rng = rng
        self.count = 0

    def update(self, x: Tensor, y: Tensor) -> None:
        batch_size = x.shape[0]
        assert x.shape == (
            batch_size,
            self.feature_count,
        )
        assert y.shape == (batch_size,)

        for i in range(batch_size):
            if self.count < self.item_count:
                # Fill the reservoir
                self.reservoir_x[self.count] = x[i]
                self.reservoir_y[self.count] = y[i]
            else:
                # Reservoir sampling
                index = torch.randint(0, self.count + 1, (1,), generator=self.rng)
                if index < self.item_count:
                    self.reservoir_x[index] = x[i]
                    self.reservoir_y[index] = y[i]
            self.count += 1

    def sample_n(self, n: int) -> Tuple[Tensor, Tensor]:
        indices = torch.randint(0, min(self.count, self.item_count), (n,))
        return self.reservoir_x[indices], self.reservoir_y[indices]


class ExperienceReplay(BatchClassifier, TaskAware):
    """Experience Replay (ER) strategy for continual learning.

    * Uses a replay buffer to store past experiences and samples from it during
      training to mitigate catastrophic forgetting.
    * The replay buffer is implemented using reservoir sampling, which allows
      for uniform sampling over the entire stream [vitter1985]_.
    * Not actually :class:`capymoa.ocl.base.TaskAware`, but will proxy it to the
      wrapped learner.

    .. [vitter1985] Jeffrey S. Vitter. 1985. Random sampling with a reservoir.
       ACM Trans. Math. Softw. 11, 1 (March 1985), 37–57.
       https://doi.org/10.1145/3147.3165
    """

    def __init__(
        self,
        learner: Finetune,
        buffer_size: int = 200,
    ) -> None:
        super().__init__(learner.schema, learner.random_seed)
        #: The wrapped learner to be trained with experience replay.
        self.learner = learner
        self._buffer = _ReservoirSampler(
            item_count=buffer_size,
            feature_count=self.schema.get_num_attributes(),
            rng=torch.Generator().manual_seed(learner.random_seed),
        )

    def batch_train(self, x: np.ndarray, y: np.ndarray) -> None:
        # preprocess the data
        x_: Tensor = torch.from_numpy(x)
        y_: Tensor = torch.from_numpy(y).long()

        # update the buffer with the new data
        self._buffer.update(x_, y_)

        # sample from the buffer and construct training batch
        replay_x, replay_y = self._buffer.sample_n(x_.shape[0])
        train_x = torch.cat((x_, replay_x), dim=0).to(self.learner.device)
        train_y = torch.cat((y_, replay_y), dim=0).to(self.learner.device)
        return self.learner.torch_batch_train(train_x, train_y)

    def batch_predict_proba(self, x: np.ndarray) -> np.ndarray:
        # preprocess the data
        x_: Tensor = torch.from_numpy(x).to(self.learner.device)
        return self.learner.torch_batch_predict_proba(x_)

    def set_test_task(self, test_task_id: int):
        if isinstance(self.learner, TaskAware):
            self.learner.set_test_task(test_task_id)

    def set_train_task(self, train_task_id: int):
        if isinstance(self.learner, TaskBoundaryAware):
            self.learner.set_train_task(train_task_id)

    def __str__(self) -> str:
        return f"ExperienceReplay(buffer_size={self._buffer.item_count})"
