"""Maximally Interfered Retrieval."""

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from capymoa.base import BatchClassifier
from capymoa.core import Instance, LabelProbabilities
from capymoa.ocl.util._replay import ReservoirSampler
from capymoa.stream import Schema


class MIR(BatchClassifier):
    """Maximally Interfered Retrieval (MIR).

    MIR [#mir]_ replays stored examples whose cross entropy increases most after
    a virtual update on the incoming batch. Candidates are drawn without
    replacement from the existing reservoir. The real update minimizes mean
    cross entropy over the incoming and selected examples together. Only then
    does the incoming batch enter memory. No task boundaries are required.

    This implementation supports SGD, including parameter groups, momentum,
    dampening, Nesterov momentum and weight decay. Retrieval uses evaluation
    mode to keep dropout and batch normalization from changing the scores or
    stored statistics. The real update uses training mode, so these layers can
    make its gradient differ from the virtual gradient. Scoring leaves model
    parameters, buffers, gradients and optimizer state unchanged.

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
        optimiser: torch.optim.SGD,
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
        :param optimiser: SGD optimizer used for both virtual and real updates.
            Other optimizer types and differentiable SGD are not supported.
        :param buffer_capacity: Maximum number of examples stored in memory.
        :param replay_batch_size: Maximum number of selected examples per update.
        :param candidate_size: Maximum number of memory examples to score.
            Must be at least ``replay_batch_size``; draws have no duplicates.
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
        if not isinstance(optimiser, torch.optim.SGD):
            raise TypeError("MIR requires torch.optim.SGD")
        if any(g.get("differentiable", False) for g in optimiser.param_groups):
            raise ValueError("Differentiable SGD is not supported")
        model_params = {id(p) for p in model.parameters()}
        optim_params = {id(p) for g in optimiser.param_groups for p in g["params"]}
        trainable = {id(p) for p in model.parameters() if p.requires_grad}
        if (
            not trainable
            or not trainable <= optim_params
            or not optim_params <= model_params
        ):
            raise ValueError("Optimizer must cover all trainable model parameters only")
        self.device = torch.device(device)
        self._model = model.to(self.device)
        self._optimiser = optimiser
        self._shape = schema.shape
        self._replay_batch_size = replay_batch_size
        self._candidate_size = candidate_size
        self._rng = torch.Generator().manual_seed(seed)
        self._buffer = ReservoirSampler(
            buffer_capacity,
            {
                "x": ((schema.get_num_attributes(),), self.x_dtype),
                "y": ((), self.y_dtype),
            },
            self._rng,
        ).to(self.device)

    def _virtual_parameters(self, x: Tensor, y: Tensor) -> dict[str, Tensor]:
        """Compute an SGD step without changing parameters or optimizer state."""
        params = dict(self._model.named_parameters())
        trainable = {name: p for name, p in params.items() if p.requires_grad}
        buffers = {name: b.clone() for name, b in self._model.named_buffers()}
        logits = torch.func.functional_call(self._model, (params, buffers), (x,))
        grads = torch.autograd.grad(
            F.cross_entropy(logits, y), tuple(trainable.values()), allow_unused=True
        )
        by_id = {id(p): g for p, g in zip(trainable.values(), grads)}
        updated = {id(p): p.detach() for p in params.values()}
        with torch.no_grad():
            for group in self._optimiser.param_groups:
                for p in group["params"]:
                    grad = by_id.get(id(p))
                    if grad is None:
                        continue
                    direction = -grad if group["maximize"] else grad
                    direction = direction + group["weight_decay"] * p
                    momentum = group["momentum"]
                    if momentum:
                        previous = self._optimiser.state.get(p, {}).get(
                            "momentum_buffer"
                        )
                        velocity = (
                            direction
                            if previous is None
                            else momentum * previous
                            + (1 - group["dampening"]) * direction
                        )
                        direction = (
                            direction + momentum * velocity
                            if group["nesterov"]
                            else velocity
                        )
                    updated[id(p)] = p - group["lr"] * direction
        return {name: updated[id(p)] for name, p in params.items()}

    def _interference_scores(
        self, x: Tensor, y: Tensor, candidate_x: Tensor, candidate_y: Tensor
    ) -> Tensor:
        modes = {module: module.training for module in self._model.modules()}
        self._model.eval()
        try:
            with torch.enable_grad():
                virtual = self._virtual_parameters(x, y)
            with torch.no_grad():
                buffers = {name: b.clone() for name, b in self._model.named_buffers()}
                before = torch.func.functional_call(
                    self._model,
                    (dict(self._model.named_parameters()), buffers),
                    (candidate_x,),
                )
                buffers = {name: b.clone() for name, b in self._model.named_buffers()}
                after = torch.func.functional_call(
                    self._model, (virtual, buffers), (candidate_x,)
                )
                return F.cross_entropy(
                    after, candidate_y, reduction="none"
                ) - F.cross_entropy(before, candidate_y, reduction="none")
        finally:
            for module, training in modes.items():
                module.training = training

    def batch_train(self, x: Tensor, y: Tensor) -> None:
        """Retrieve historical examples, learn, then update the reservoir."""
        x = x.to(self.device, self.x_dtype).reshape(
            -1, self.schema.get_num_attributes()
        )
        y = y.to(self.device, self.y_dtype)
        current_x = x.reshape(-1, *self._shape)
        train_x, train_y = current_x, y
        if self._buffer.count:
            indices = torch.randperm(self._buffer.count, generator=self._rng)[
                : self._candidate_size
            ].to(self.device)
            memory = self._buffer.array()
            candidate_x = memory["x"][indices].reshape(-1, *self._shape)
            candidate_y = memory["y"][indices]
            scores = self._interference_scores(current_x, y, candidate_x, candidate_y)
            selected = scores.topk(min(self._replay_batch_size, len(scores))).indices
            train_x = torch.cat((current_x, candidate_x[selected]))
            train_y = torch.cat((y, candidate_y[selected]))
        self._model.train()
        self._optimiser.zero_grad()
        loss = F.cross_entropy(self._model(train_x), train_y)
        loss.backward()
        self._optimiser.step()
        self._buffer.update(x=x, y=y)

    @torch.no_grad()
    def batch_predict_proba(self, x: Tensor) -> Tensor:
        """Return class probabilities for schema-shaped or flat examples."""
        modes = {module: module.training for module in self._model.modules()}
        self._model.eval()
        try:
            x = x.to(self.device, self.x_dtype).reshape(-1, *self._shape)
            return self._model(x).softmax(dim=1)
        finally:
            for module, training in modes.items():
                module.training = training

    def predict_proba(self, instance: Instance) -> LabelProbabilities:
        """Return single-instance probabilities on CPU."""
        x = torch.as_tensor(instance.x).reshape(1, -1)
        return (
            self.batch_predict_proba(x)
            .to(device="cpu", dtype=torch.float64)
            .numpy()
            .reshape(-1)
        )

    def __str__(self) -> str:
        return f"MIR(buffer_capacity={self._buffer.capacity}, candidate_size={self._candidate_size})"
