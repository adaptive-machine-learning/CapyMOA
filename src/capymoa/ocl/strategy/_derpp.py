"""Dark Experience Replay++ built on the existing DER strategy."""

from collections.abc import Callable
from math import isfinite

import torch
from torch import Tensor, nn

from capymoa.core import Instance, LabelProbabilities
from capymoa.stream import Schema

from ._der import DER


class DERPP(DER):
    """Dark Experience Replay++ (DER++).

    DER++ [#derpp]_ extends :class:`DER` with cross entropy on a second, independently sampled
    replay batch. The objective is current-label cross entropy plus ``alpha``
    times historical-logit MSE plus ``beta`` times replay-label cross entropy.
    All terms contribute to the same optimizer step. Setting ``beta=0`` uses
    DER's training step directly.

    The model, optimizer, schema shaping, augmentation, batch loop and named
    replay buffer are inherited from DER. Sampling is with replacement and the
    buffer stays on the training device. Inputs, labels and pre-update logits
    enter memory on the first substep, after both replay draws. As in DER, later
    substeps can replay the newly inserted examples. Use ``substeps=1`` for one
    update using only examples from earlier batches. No task boundaries are needed.

    >>> from capymoa.ocl.strategy import DERPP
    >>> from capymoa.stream import Schema
    >>> import torch
    >>> from torch import nn
    >>> schema = Schema.from_custom(
    ...     ["x0", "x1", "class"], "class", {"class": ["a", "b"]}
    ... )
    >>> model = nn.Linear(2, 2)
    >>> learner = DERPP(schema, model, torch.optim.SGD(model.parameters(), lr=0.01))
    >>> learner.batch_train(torch.eye(2), torch.tensor([0, 1]))
    >>> learner.batch_predict_proba(torch.ones(3, 2)).shape
    torch.Size([3, 2])

    .. [#derpp] Buzzega, Pietro, Matteo Boschini, Angelo Porrello, Davide Abati,
        and Simone Calderara. "Dark Experience for General Continual Learning:
        A Strong, Simple Baseline." NeurIPS 2020.
        https://papers.nips.cc/paper/2020/hash/b704ea2c39778f07c617f6b7ce480e9e-Abstract.html
    """

    def __init__(
        self,
        schema: Schema,
        model: nn.Module,
        optimiser: torch.optim.Optimizer,
        augment: Callable[[Tensor], Tensor] | None = None,
        alpha: float = 0.5,
        buffer_capacity: int = 200,
        seed: int = 0,
        substeps: int = 1,
        device: torch.device | str = "cpu",
        *,
        beta: float = 0.5,
    ) -> None:
        """Construct a DER++ learner.

        :param schema: Classification schema with fixed features and classes.
        :param model: Network returning one logit per schema class. Receives
            batches shaped as ``(batch_size, *schema.shape)``. The caller seeds
            model initialization, as for DER.
        :param optimiser: Optimizer for ``model``, using DER's constructor API.
        :param augment: Augmentation of schema-shaped batches; defaults to none.
        :param alpha: Finite non-negative weight of historical-logit MSE.
        :param buffer_capacity: Maximum number of examples, labels and logits.
        :param seed: Seed of DER's private replay generator.
        :param substeps: Optimization steps per incoming batch. Memory is updated
            once; later steps can replay examples from that incoming batch.
        :param device: Device for the network and replay buffer.
        :param beta: Finite non-negative replay-label CE weight. Zero recovers DER.
        """
        if not schema.is_classification():
            raise ValueError("DERPP requires a classification schema")
        if not isfinite(alpha) or alpha < 0:
            raise ValueError("alpha must be finite and non-negative")
        if not isfinite(beta) or beta < 0:
            raise ValueError("beta must be finite and non-negative")
        if not isinstance(buffer_capacity, int) or buffer_capacity <= 0:
            raise ValueError("buffer_capacity must be a positive integer")
        if not isinstance(substeps, int) or substeps <= 0:
            raise ValueError("substeps must be a positive integer")
        super().__init__(
            schema,
            model,
            optimiser,
            augment,
            alpha,
            buffer_capacity,
            seed,
            substeps,
            device,
        )
        self._beta = float(beta)

    def _train_step(self, x: Tensor, y: Tensor, update_buffer: bool) -> None:
        if self._beta == 0:
            return super()._train_step(x, y, update_buffer)

        self._optimiser.zero_grad()
        n = x.shape[0]
        z = self._model(self._augment(x.view(-1, *self._shape)))
        # Preserve targets even if the model returns a view of mutable parameters.
        stored_z = z.detach().clone()
        loss = self._criterion(z, y)
        if self._buffer.count > 0:
            dark_replay = self._buffer.sample(n)
            xp = dark_replay["x"].to(self.device).view(-1, *self._shape)
            zp = dark_replay["z"].to(self.device)
            loss = loss + self._alpha * self._logit_loss(
                self._model(self._augment(xp)), zp
            )
            label_replay = self._buffer.sample(n)
            xp = label_replay["x"].to(self.device).view(-1, *self._shape)
            yp = label_replay["y"].to(self.device)
            loss = loss + self._beta * self._criterion(
                self._model(self._augment(xp)), yp
            )
        if not torch.isfinite(loss):
            raise ValueError("DERPP loss is not finite")
        if update_buffer:
            self._buffer.update(x=x, z=stored_z, y=y)
        loss.backward()
        self._optimiser.step()

    def predict_proba(self, instance: Instance) -> LabelProbabilities:
        """Return single-instance probabilities on CPU, including for GPU models."""
        x = torch.as_tensor(instance.x).reshape(1, -1)
        return (
            self.batch_predict_proba(x)
            .to(device="cpu", dtype=torch.float64)
            .numpy()
            .reshape(-1)
        )

    def __str__(self) -> str:
        return (
            f"DERPP(alpha={self._alpha}, beta={self._beta}, "
            f"buffer_capacity={self._buffer.capacity})"
        )
