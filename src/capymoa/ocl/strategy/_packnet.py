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


def get_visible_mask(allocation: Tensor, task_id: int) -> Tensor:
    """Get a mask of the weights visible to a task.

    Weights from the task and all earlier tasks are visible.

    >>> allocation = torch.tensor([0, 0, 1, 1, 2, 2], dtype=torch.uint8)
    >>> get_visible_mask(allocation, 0)
    tensor([ True,  True, False, False, False, False])
    >>> get_visible_mask(allocation, 1)
    tensor([ True,  True,  True,  True, False, False])

    :param allocation: The task that owns each weight.
    :param task_id: The task ID.
    :return: A boolean mask.
    """
    return (allocation <= task_id).bool()


def get_task_mask(allocation: Tensor, task_id: int) -> Tensor:
    """Get a mask of the weights owned by a task.

    >>> allocation = torch.tensor([0, 0, 1, 1], dtype=torch.uint8)
    >>> get_task_mask(allocation, 0)
    tensor([ True,  True, False, False])
    >>> get_task_mask(allocation, 1)
    tensor([False, False,  True,  True])

    :param allocation: The task that owns each weight.
    :param task_id: The task ID.
    :return: A boolean mask.
    """
    return (allocation == task_id).bool()


def disable_grads(allocation: Tensor, grads: Tensor, task_id: int) -> Tensor:
    """Zero the gradients of weights that a task does not own.

    >>> allocation = torch.tensor([0, 1], dtype=torch.uint8)
    >>> disable_grads(allocation, torch.ones(2), 1)
    tensor([0., 1.])

    :param allocation: The task that owns each weight.
    :param grads: Gradients with the same number of elements as ``allocation``.
    :param task_id: The task being trained.
    :return: Masked gradients, with the shape of ``grads``.
    """
    return (grads.view(-1) * get_task_mask(allocation, task_id)).view_as(grads)


def update_allocation(
    allocation: Tensor,
    src_prune_order: Tensor,
    target_src_sparsity: float,
    src_task_id: int,
    dst_task_id: int,
) -> Tensor:
    """Move the least important weights from one task to another.

    >>> allocation = torch.tensor([0, 0, 0, 0, 0, 0, 1, 1], dtype=torch.uint8)
    >>> update_allocation(allocation.clone(), torch.arange(6), 0.5, 0, 1)
    tensor([1, 1, 0, 0, 0, 0, 1, 1], dtype=torch.uint8)
    >>> update_allocation(allocation.clone(), torch.arange(6), 0.75, 0, 1)
    tensor([1, 1, 1, 1, 0, 0, 1, 1], dtype=torch.uint8)

    :param allocation: The task that owns each weight. Updated in place.
    :param src_prune_order: Indices into the source task's weights, least important
        first.
    :param target_src_sparsity: Fraction of the source and destination weights that
        should not belong to the source after the move.
    :param src_task_id: The task that gives weights.
    :param dst_task_id: The task that receives weights.
    :return: The updated allocation.
    """
    allocation = allocation.view(-1)
    src_prune_order = src_prune_order.view(-1)
    assert 0 <= target_src_sparsity <= 1, "The sparsity must be between 0 and 1"
    if src_prune_order.numel() == 0:
        return allocation

    src_mask = get_task_mask(allocation, src_task_id)
    src_count = int(src_mask.sum())
    dst_count = int(get_task_mask(allocation, dst_task_id).sum())
    assert len(src_prune_order) == src_count, "Order must cover the source weights"

    # Number of weights the source keeps
    keep = math.ceil((src_count + dst_count) * (1 - target_src_sparsity))
    n_move = src_count - keep
    if n_move <= 0:
        return allocation

    allocation[src_mask] = allocation[src_mask].index_fill(
        0, src_prune_order[:n_move], dst_task_id
    )
    return allocation


def prune_by_abs_magnitude(
    allocation: Tensor, parameter: Tensor, task_id: int, sparsity: float
) -> Tensor:
    """Move the smallest weights of ``task_id`` to ``task_id + 1``.

    >>> allocation = torch.zeros(4, dtype=torch.uint8)
    >>> prune_by_abs_magnitude(allocation, torch.tensor([3.0, -1, 2, -4]), 0, 0.5)
    tensor([0, 1, 1, 0], dtype=torch.uint8)

    :param allocation: The task that owns each weight. Updated in place.
    :param parameter: The weights.
    :param task_id: The task to prune.
    :param sparsity: Fraction of the weights to move.
    :return: The updated allocation.
    """
    values = parameter.detach().view(-1)[get_task_mask(allocation, task_id)]
    order = torch.argsort(values.abs())
    return update_allocation(allocation, order, sparsity, task_id, task_id + 1)


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
                f"alloc_{i}", torch.zeros(param.numel(), dtype=torch.uint8)
            )

    def allocations(self) -> list[Tensor]:
        return [getattr(self, f"alloc_{i}") for i in range(len(self.names))]

    def forward(self, x: Tensor, task: int) -> Tensor:
        """Run ``model`` using only the weights visible to ``task``."""
        params = {}
        for name, alloc in zip(self.names, self.allocations()):
            p = self.model.get_parameter(name)
            params[name] = (p.view(-1) * get_visible_mask(alloc, task)).view_as(p)
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
        allocs = module.allocations()
        saved = [p.detach().clone() for p in params]
        for p, alloc in zip(params, allocs):
            if p.grad is not None:
                p.grad = disable_grads(alloc, p.grad, task)
        self._optimiser.step()
        # Weight decay and momentum can move frozen weights, so put them back.
        with torch.no_grad():
            for p, alloc, old in zip(params, allocs, saved):
                own = get_task_mask(alloc, task).view_as(p)
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
        for name, alloc in zip(module.names, module.allocations()):
            prune_by_abs_magnitude(
                alloc, module.model.get_parameter(name), task, sparsity
            )

    def __str__(self) -> str:
        return f"PackNet(sparsity={self._sparsity})"
