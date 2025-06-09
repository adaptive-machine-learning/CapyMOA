from typing import Tuple

import torch
from torch import Tensor

from capymoa.base import BatchClassifier
from capymoa.ocl.base import TrainTaskAware, TestTaskAware


class _ReservoirSampler:
    def __init__(self, item_count: int, feature_count: int, rng: torch.Generator):
        self.max_count = item_count
        self.count = 0
        self.in_features = feature_count
        self.reservoir_x = torch.zeros((item_count, feature_count))
        self.reservoir_y = torch.zeros((item_count,), dtype=torch.long)
        self.rng = rng

    def update(self, x: Tensor, y: Tensor) -> None:
        x = x.to(self.reservoir_x.device)
        y = y.to(self.reservoir_y.device)
        batch_size = x.shape[0]
        assert x.shape == (
            batch_size,
            self.in_features,
        )
        assert y.shape == (batch_size,)

        for i in range(batch_size):
            if self.count < self.max_count:
                # Fill the reservoir
                self.reservoir_x[self.count] = x[i]
                self.reservoir_y[self.count] = y[i]
            else:
                # Reservoir sampling
                index = torch.randint(0, self.count + 1, (1,), generator=self.rng)
                if index < self.max_count:
                    self.reservoir_x[index] = x[i]
                    self.reservoir_y[index] = y[i]
            self.count += 1

    @property
    def is_empty(self) -> bool:
        return self.count == 0

    def sample_n(self, n: int) -> Tuple[Tensor, Tensor]:
        indices = torch.randint(0, min(self.count, self.max_count), (n,))
        return self.reservoir_x[indices], self.reservoir_y[indices]


class ExperienceReplay(BatchClassifier, TrainTaskAware, TestTaskAware):
    """Experience Replay (ER) strategy for continual learning.

    * Uses a replay buffer to store past experiences and samples from it during
      training to mitigate catastrophic forgetting.
    * The replay buffer is implemented using reservoir sampling, which allows
      for uniform sampling over the entire stream [vitter1985]_.
    * Not :class:`capymoa.ocl.base.TrainTaskAware` or
      :class:`capymoa.ocl.base.TestTaskAware`, but will proxy it to the wrapped
      learner.

    .. [vitter1985] Jeffrey S. Vitter. 1985. Random sampling with a reservoir.
       ACM Trans. Math. Softw. 11, 1 (March 1985), 37–57.
       https://doi.org/10.1145/3147.3165
    """

    def __init__(
        self, learner: BatchClassifier, buffer_size: int = 200, repeat: int = 1
    ) -> None:
        """Initialize the Experience Replay strategy.

        :param learner: The learner to be wrapped for experience replay.
        :param buffer_size: The size of the replay buffer, defaults to 200.
        :param repeat: The number of times to repeat the training data in each batch,
            defaults to 1.
        """
        super().__init__(learner.schema, learner.random_seed)
        #: The wrapped learner to be trained with experience replay.
        self.learner = learner
        self._buffer = _ReservoirSampler(
            item_count=buffer_size,
            feature_count=self.schema.get_num_attributes(),
            rng=torch.Generator().manual_seed(learner.random_seed),
        )
        self.repeat = repeat

    def batch_train(self, x: Tensor, y: Tensor) -> None:
        # update the buffer with the new data
        self._buffer.update(x, y)

        for _ in range(self.repeat):
            # sample from the buffer and construct training batch
            replay_x, replay_y = self._buffer.sample_n(x.shape[0])
            train_x = torch.cat((x, replay_x), dim=0)
            train_y = torch.cat((y, replay_y), dim=0)
            train_x = train_x.to(self.learner.device, dtype=self.learner.x_dtype)
            train_y = train_y.to(self.learner.device, dtype=self.learner.y_dtype)
            self.learner.batch_train(train_x, train_y)

    def batch_predict_proba(self, x: Tensor) -> Tensor:
        x = x.to(self.learner.device, dtype=self.learner.x_dtype)
        return self.learner.batch_predict_proba(x)

    def on_test_task(self, task_id: int):
        if isinstance(self.learner, TestTaskAware):
            self.learner.on_test_task(task_id)

    def on_train_task(self, task_id: int):
        if isinstance(self.learner, TrainTaskAware):
            self.learner.on_train_task(task_id)

    def __str__(self) -> str:
        return f"ExperienceReplay(buffer_size={self._buffer.max_count})"
