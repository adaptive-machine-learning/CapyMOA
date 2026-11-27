import math

import torch
from torch import Tensor, nn
from torch.func import functional_call

from capymoa.base import BatchClassifier, TorchMixin
from capymoa.ocl.evaluate.events import TestTaskBegin, TrainTaskBegin, TrainTaskEnd
from capymoa.ocl.events import Dispatcher, Handler
from capymoa.ocl.util._optim import reset_optimizer_state
from capymoa.stream._stream import Schema

NEG_INF = float("-inf")


class _PackNetModule(nn.Module):
    """Torch state of PackNet."""

    def __init__(self, model: nn.Module, task_mask: Tensor) -> None:
        super().__init__()
        self.model = model
        self.criterion = nn.CrossEntropyLoss()
        self.task_mask = nn.Buffer(task_mask)
        self.names = [n for n, p in model.named_parameters() if p.requires_grad]
        # ``alloc_i[j] = t`` means task ``t`` owns weight ``j``. Weights start with
        # task 0 and move to later tasks when pruned.
        for i, name in enumerate(self.names):
            param = model.get_parameter(name)
            self.register_buffer(
                f"alloc_{i}", torch.zeros_like(param, dtype=torch.uint8)
            )

    def allocations(self) -> list[Tensor]:
        return [getattr(self, f"alloc_{i}") for i in range(len(self.names))]

    def forward(self, x: Tensor, task: int) -> Tensor:
        """Run ``model`` using only the weights visible to ``task``."""
        params = {
            n: self.model.get_parameter(n) * (a <= task)
            for n, a in zip(self.names, self.allocations())
        }
        return functional_call(self.model, params, (x,), strict=False)


class PackNet(TorchMixin[_PackNetModule], BatchClassifier, Handler):
    """Task-incremental PackNet.

    PackNet [#f1]_ gives each task its own subset of the weights. While training task
    ``t``, only weights owned by ``t`` change, and the model sees weights owned by
    tasks ``0..t``. When task ``t`` ends, the smallest weights (by magnitude) of ``t``
    are freed for task ``t + 1``. Old tasks cannot be forgotten, since their weights
    never change. Task identity is needed at test time, and is used to mask the
    output logits to the classes of that task.

    The allocation follows the ``PrunableParameter`` in
    https://github.com/tachyonicClock/AAAI25-clkan.

    Batch norm layers are put in eval mode after the first prune, so their running
    statistics stay fixed. Other buffers are shared by all tasks, and not protected.

    ..  [#f1] Mallya, A., & Lazebnik, S. (2018). PackNet: Adding multiple tasks to a
        single network by iterative pruning. CVPR. https://arxiv.org/abs/1711.05769
    """

    def __init__(
        self,
        schema: Schema,
        model: nn.Module,
        optimiser: torch.optim.Optimizer,
        task_mask: Tensor,
        sparsity: float | None = None,
        device: torch.device | str = "cpu",
    ) -> None:
        """Construct a PackNet learner.

        :param schema: Describes the stream's structure and types.
        :param model: Torch model that outputs class logits.
        :param optimiser: Optimiser for ``model``'s parameters.
        :param task_mask: Boolean tensor of shape ``(n_tasks, n_classes)`` that marks
            the classes of each task.
        :param sparsity: Fraction of a task's weights to free for later tasks when
            the task ends. If ``None``, the remaining weights are shared equally
            between the remaining tasks.
        :param device: Compute device.
        :raises ValueError: If ``sparsity`` is not in ``[0, 1]``.
        """
        super().__init__(schema, 0)
        if sparsity is not None and not 0 <= sparsity <= 1:
            raise ValueError("sparsity must be between 0 and 1.")
        self.x_dtype = next(model.parameters(), torch.empty(0)).dtype
        self._optimiser = optimiser
        self._sparsity = sparsity
        self._train_task = 0
        self._test_task = 0
        self._bn_frozen = False
        self._module = _PackNetModule(model, task_mask).to(device)

    @property
    def n_tasks(self) -> int:
        """Number of tasks."""
        return self._module.task_mask.shape[0]

    def batch_train(self, x: Tensor, y: Tensor) -> None:
        module, task = self._module, self._train_task
        module.train()
        if self._bn_frozen:
            for m in module.modules():
                if isinstance(m, nn.modules.batchnorm._BatchNorm):
                    m.eval()
        self._optimiser.zero_grad()
        logits = self._mask_logits(module(x, task), task)
        module.criterion(logits, y).backward()

        params = [module.model.get_parameter(n) for n in module.names]
        owned = [a == task for a in module.allocations()]
        saved = [p.detach().clone() for p in params]
        for p, own in zip(params, owned):
            if p.grad is not None:
                p.grad.mul_(own)
        self._optimiser.step()
        # Weight decay and momentum can move frozen weights, so put them back.
        with torch.no_grad():
            for p, own, old in zip(params, owned, saved):
                p.copy_(torch.where(own, p, old))

    @torch.no_grad()
    def batch_predict_proba(self, x: Tensor) -> Tensor:
        self._module.eval()
        logits = self._module(x, self._test_task)
        return torch.softmax(self._mask_logits(logits, self._test_task), dim=1)

    def attach_with(self, source: Dispatcher) -> "PackNet":
        source.subscribe(TrainTaskBegin, self._on_train_task_begin)
        source.subscribe(TrainTaskEnd, self._on_train_task_end)
        source.subscribe(TestTaskBegin, self._on_test_task_begin)
        return self

    def _on_train_task_begin(self, event: TrainTaskBegin) -> None:
        reset_optimizer_state(self._optimiser)
        self._train_task = event.train_task

    def _on_train_task_end(self, event: TrainTaskEnd) -> None:
        self._prune(event.train_task)

    def _on_test_task_begin(self, event: TestTaskBegin) -> None:
        self._test_task = event.test_task

    def _mask_logits(self, logits: Tensor, task: int) -> Tensor:
        return logits.masked_fill(self._module.task_mask[task] == 0, NEG_INF)

    @torch.no_grad()
    def _prune(self, task: int) -> None:
        """Move the smallest weights of ``task`` to ``task + 1``."""
        self._bn_frozen = True
        remaining = self.n_tasks - task
        if remaining <= 1:
            return
        sparsity = self._sparsity
        if sparsity is None:
            sparsity = 1 - 1 / remaining

        module = self._module
        params = [module.model.get_parameter(n) for n in module.names]
        allocs = module.allocations()
        owned = [a == task for a in allocs]
        if not any(o.any() for o in owned):
            return
        # Pool weights from all tensors, so pruning is global.
        values = torch.cat([p.detach().abs()[o] for p, o in zip(params, owned)])
        n_prune = math.floor(values.numel() * sparsity)
        if n_prune == 0:
            return
        order = torch.argsort(values)[:n_prune]
        prune = torch.zeros(values.numel(), dtype=torch.bool, device=values.device)
        prune[order] = True
        start = 0
        for a, o in zip(allocs, owned):
            n = int(o.sum())
            a[o] = torch.where(prune[start : start + n], task + 1, task).to(a.dtype)
            start += n

    def __str__(self) -> str:
        return f"PackNet(sparsity={self._sparsity})"
