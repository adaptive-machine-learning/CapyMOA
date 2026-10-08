from collections.abc import Callable

import torch
from torch import Tensor

from capymoa.base import BatchClassifier
from capymoa.ocl.util._replay import ReplayBuffer, ReservoirSampler
from capymoa.stream import Schema


class DER(BatchClassifier):
    """Dark Experience Replay.

    Dark Experience Replay (DER) [#f1]_ is a replay strategy. It stores the model
    logits of replay samples. It adds an MSE loss between the stored logits and the
    new logits for those samples.

    ..  [#f1] Buzzega, Pietro, Matteo Boschini, Angelo Porrello, Davide Abati,
        and Simone Calderara. "Dark Experience for General Continual Learning:
        A Strong, Simple Baseline." Advances in Neural Information Processing
        Systems 33 (2020): 15920–30.
        https://papers.nips.cc/paper/2020/hash/b704ea2c39778f07c617f6b7ce480e9e-Abstract.html.
    """

    def __init__(
        self,
        schema: Schema,
        model: torch.nn.Module,
        optimiser: torch.optim.Optimizer,
        augment: Callable[[Tensor], Tensor] | None = None,
        alpha: float = 0.5,
        buffer_capacity: int = 200,
        replay_buffer: ReplayBuffer | None = None,
        seed: int = 0,
        substeps: int = 1,
        device: torch.device | str = "cpu",
    ) -> None:
        """Construct a DER learner.

        :param schema: Stream schema.
        :param model: Torch model that outputs class logits.
        :param optimiser: Optimiser for the parameters of ``model``.
        :param augment: Augmentation for a batch of examples shaped like
            ``schema.shape``. Defaults to no augmentation.
        :param alpha: Weight of the logit replay loss.
        :param buffer_capacity: Number of replay samples to keep.
        :param replay_buffer: Replay buffer with keys ``x``, ``z`` and ``y``. By
            default, a reservoir sampler of size ``buffer_capacity``.
        :param seed: Random seed.
        :param substeps: Optimisation steps per batch. Each step uses a new random
            augmentation of the batch and the replay samples.
        :param device: Compute device.
        """
        super().__init__(schema, seed)
        if alpha < 0:
            raise ValueError("alpha must be non-negative.")
        if buffer_capacity <= 0:
            raise ValueError("buffer_capacity must be greater than zero.")
        if substeps <= 0:
            raise ValueError("substeps must be greater than zero.")

        self.device = torch.device(device)
        self._augment = augment if augment is not None else (lambda x: x)
        self._alpha = alpha
        self._substeps = substeps
        self._model = model.to(self.device)
        self._optimiser = optimiser
        self._criterion = torch.nn.CrossEntropyLoss()
        self._logit_loss = torch.nn.MSELoss()
        self._shape = schema.shape
        if replay_buffer is None:
            replay_buffer = ReservoirSampler(
                buffer_capacity,
                {
                    "x": ((schema.get_num_attributes(),), torch.float32),
                    "z": ((schema.get_num_classes(),), torch.float32),
                    "y": ((), torch.long),
                },
                torch.Generator().manual_seed(seed),
            )
        self._buffer = replay_buffer.to(self.device)

    def _train_step(self, x: Tensor, y: Tensor, update_buffer: bool) -> None:
        self._optimiser.zero_grad()
        n = x.shape[0]

        z = self._model(self._augment(x.view(-1, *self._shape)))
        loss = self._criterion(z, y)
        # Sample before the update, so a batch is not replayed against itself.
        if self._buffer.count > 0:
            replay = self._buffer.sample(n)
            xp = replay["x"].to(self.device).view(-1, *self._shape)
            zp = replay["z"].to(self.device)
            loss = loss + self._alpha * self._logit_loss(
                self._model(self._augment(xp)), zp
            )
        if update_buffer:
            self._buffer.update(x=x, z=z, y=y)
        loss.backward()
        self._optimiser.step()

    def batch_train(self, x: Tensor, y: Tensor) -> None:
        """Train on a batch (Algorithm 1 of the DER paper)."""
        x = x.to(self.device, self.x_dtype)
        y = y.to(self.device, self.y_dtype)
        self._model.train()
        for i in range(self._substeps):
            self._train_step(x, y, update_buffer=i == 0)

    @torch.no_grad()
    def batch_predict_proba(self, x: Tensor) -> Tensor:
        self._model.eval()
        x = x.to(self.device, self.x_dtype).view(-1, *self._shape)
        return torch.softmax(self._model(x), dim=1)

    def __str__(self) -> str:
        return f"DER(alpha={self._alpha}, buffer_capacity={self._buffer.capacity})"
