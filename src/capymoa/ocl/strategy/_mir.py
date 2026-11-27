"""Maximally Interfered Retrieval."""

import copy
from collections.abc import Iterator
from contextlib import contextmanager

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from capymoa.base import BatchClassifier, TorchMixin
from capymoa.ocl.util._replay import ReservoirSampler
from capymoa.stream import Schema


@contextmanager
def _eval_mode(model: nn.Module) -> Iterator[None]:
    """Switch ``model`` to evaluation mode, then restore each module's mode."""
    modes = {module: module.training for module in model.modules()}
    model.eval()
    try:
        yield
    finally:
        for module, training in modes.items():
            module.training = training


class _MIRModule(nn.Module):
    """Torch state of MIR."""

    def __init__(self, model: nn.Module, buffer: ReservoirSampler) -> None:
        super().__init__()
        self.model = model
        self.buffer = buffer


class MIR(TorchMixin[_MIRModule], BatchClassifier):
    """Maximally Interfered Retrieval (MIR,2019).

    MIR [#mir]_ replays stored examples whose cross entropy increases most after
    a virtual update on the incoming batch. Candidates are sampled from the
    reservoir. The real update minimizes mean cross entropy over the incoming and
    selected examples together. Only then does the incoming batch enter memory.
    No task boundaries are required.

    Any PyTorch optimizer works. The virtual update takes one real optimizer step,
    then restores parameters, gradients and optimizer state. Retrieval uses
    evaluation mode to keep dropout and batch normalization from changing the
    scores or stored statistics. The real update uses training mode, so these
    layers can make its gradient differ from the virtual gradient.

    >>> from capymoa.ocl.strategy import MIR
    >>> schema = Schema.from_custom(
    ...     ["x0", "x1", "class"], "class", {"class": ["a", "b"]}
    ... )
    >>> model = nn.Linear(2, 2)
    >>> learner = MIR(schema, model, torch.optim.SGD(model.parameters(), lr=0.1))
    >>> learner.batch_train(torch.eye(2), torch.tensor([0, 1]))
    >>> learner.batch_train(torch.eye(2), torch.tensor([1, 0]))
    >>> learner.batch_predict_proba(torch.ones(3, 2)).shape
    torch.Size([3, 2])

    .. [#mir] Aljundi et al. "Online Continual Learning with Maximally
        Interfered Retrieval." Advances in Neural Information Processing
        Systems 32 (2019).
        https://arxiv.org/abs/1908.04742
    """

    def __init__(
        self,
        schema: Schema,
        model: nn.Module,
        optimiser: torch.optim.Optimizer,
        buffer_capacity: int = 200,
        replay_batch_size: int = 32,
        candidate_size: int = 128,
        seed: int = 0,
        device: torch.device | str = "cpu",
    ) -> None:
        """Construct a MIR learner.

        :param schema: Classification schema; inputs use ``schema.shape``.
        :param model: Network returning one logit per class. The caller seeds
            model initialization and constructs the optimizer for this model.
        :param optimiser: Optimizer used for both virtual and real updates.
        :param buffer_capacity: Maximum number of examples stored in memory.
        :param replay_batch_size: Maximum number of selected examples per update.
        :param candidate_size: Maximum number of memory examples to score.
            Must be at least ``replay_batch_size``. Candidates are drawn with
            replacement.
        :param seed: Seed for reservoir updates and candidate sampling.
        :param device: Device for the model and replay memory.
        """
        super().__init__(schema, seed)
        if not schema.is_classification():
            raise ValueError("MIR requires a classification schema")
        for name, value in (
            ("buffer_capacity", buffer_capacity),
            ("replay_batch_size", replay_batch_size),
            ("candidate_size", candidate_size),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if candidate_size < replay_batch_size:
            raise ValueError("candidate_size must be at least replay_batch_size")
        self.x_dtype = next(model.parameters(), torch.empty(0)).dtype
        self._optimiser = optimiser
        self._shape = schema.shape
        self._replay_batch_size = replay_batch_size
        self._candidate_size = candidate_size
        self._rng = torch.Generator().manual_seed(seed)
        buffer = ReservoirSampler(
            buffer_capacity,
            {
                "x": ((schema.get_num_attributes(),), self.x_dtype),
                "y": ((), self.y_dtype),
            },
            self._rng,
        )
        self._module = _MIRModule(model, buffer).to(device)

    def _interference_scores(
        self, x: Tensor, y: Tensor, candidate_x: Tensor, candidate_y: Tensor
    ) -> Tensor:
        """Score candidates by loss increase after one step on ``x`` and ``y``.

        Parameters, gradients and optimizer state are restored afterwards.
        """
        params = [p for g in self._optimiser.param_groups for p in g["params"]]
        weights = [p.detach().clone() for p in params]
        grads = [None if p.grad is None else p.grad.clone() for p in params]
        state = copy.deepcopy(self._optimiser.state_dict())

        def loss() -> Tensor:
            return F.cross_entropy(
                self._module.model(candidate_x), candidate_y, reduction="none"
            )

        try:
            with _eval_mode(self._module.model):
                with torch.no_grad():
                    before = loss()
                self._optimiser.zero_grad()
                F.cross_entropy(self._module.model(x), y).backward()
                self._optimiser.step()
                with torch.no_grad():
                    after = loss()
            return after - before
        finally:
            with torch.no_grad():
                for p, w, g in zip(params, weights, grads):
                    p.copy_(w)
                    p.grad = g
            self._optimiser.load_state_dict(state)

    def batch_train(self, x: Tensor, y: Tensor) -> None:
        """Retrieve historical examples, learn, then update the reservoir."""
        x = x.reshape(-1, self.schema.get_num_attributes())
        current_x = x.reshape(-1, *self._shape)
        train_x, train_y = current_x, y
        if self._module.buffer.count:
            n = min(self._candidate_size, self._module.buffer.count)
            candidates = self._module.buffer.sample(n)
            candidate_x = candidates["x"].reshape(-1, *self._shape)
            candidate_y = candidates["y"]
            scores = self._interference_scores(current_x, y, candidate_x, candidate_y)
            selected = scores.topk(min(self._replay_batch_size, len(scores))).indices
            train_x = torch.cat((current_x, candidate_x[selected]))
            train_y = torch.cat((y, candidate_y[selected]))
        self._module.model.train()
        self._optimiser.zero_grad()
        loss = F.cross_entropy(self._module.model(train_x), train_y)
        loss.backward()
        self._optimiser.step()
        self._module.buffer.update(x=x, y=y)

    @torch.no_grad()
    def batch_predict_proba(self, x: Tensor) -> Tensor:
        """Return class probabilities for schema-shaped or flat examples."""
        with _eval_mode(self._module.model):
            x = x.reshape(-1, *self._shape)
            return self._module.model(x).softmax(dim=1)

    def __str__(self) -> str:
        return f"MIR(buffer_capacity={self._module.buffer.capacity}, candidate_size={self._candidate_size})"
