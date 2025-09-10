import torch
from torch import Tensor

from capymoa.base import BatchClassifier
from capymoa.ocl.base import TrainTaskAware, TestTaskAware
from capymoa.ocl.util._coreset import ReservoirSampler


class ExperienceReplay(BatchClassifier, TrainTaskAware, TestTaskAware):
    """Experience Replay.

    Experience Replay (ER) [#f0]_ is a replay continual learning strategy.

    * Uses a replay buffer to store past experiences and samples from it during training
      to mitigate catastrophic forgetting.
    * The replay buffer is implemented using reservoir sampling, which allows for
      uniform sampling over the entire stream [#f1]_.
    * Not :class:`capymoa.ocl.base.TrainTaskAware` or
      :class:`capymoa.ocl.base.TestTaskAware`, but will proxy it to the wrapped learner.

    >>> from capymoa.ann import Perceptron
    >>> from capymoa.classifier import Finetune
    >>> from capymoa.ocl.strategy import ExperienceReplay
    >>> from capymoa.ocl.datasets import TinySplitMNIST
    >>> from capymoa.ocl.evaluation import ocl_train_eval_loop
    >>> import torch
    >>> _ = torch.manual_seed(0)
    >>> scenario = TinySplitMNIST()
    >>> model = Perceptron(scenario.schema)
    >>> learner = ExperienceReplay(Finetune(scenario.schema, model))
    >>> results = ocl_train_eval_loop(
    ...     learner,
    ...     scenario.train_loaders(32),
    ...     scenario.test_loaders(32),
    ... )
    >>> print(f"{results.accuracy_final*100:.1f}%")
    33.0%

    .. [#f0] `Rolnick, D., Ahuja, A., Schwarz, J., Lillicrap, T., & Wayne, G. (2019).
              Experience replay for continual learning. Advances in neural information
              processing systems, 32. <https://arxiv.org/abs/1811.11682>`_
    .. [#f1] `Jeffrey S. Vitter. 1985. Random sampling with a reservoir. ACM Trans. Math.
              Softw. 11, 1 (March 1985), 37–57. <https://doi.org/10.1145/3147.3165>`_
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
        self._buffer = ReservoirSampler(
            capacity=buffer_size,
            features=self.schema.get_num_attributes(),
            rng=torch.Generator().manual_seed(learner.random_seed),
        )
        self.repeat = repeat

    def batch_train(self, x: Tensor, y: Tensor) -> None:
        # update the buffer with the new data
        self._buffer.update(x, y)

        for _ in range(self.repeat):
            # sample from the buffer and construct training batch
            replay_x, replay_y = self._buffer.sample(x.shape[0])
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
        return f"ExperienceReplay(buffer_size={self._buffer.capacity})"
