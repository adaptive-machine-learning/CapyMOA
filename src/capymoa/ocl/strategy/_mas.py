from typing import Callable, Iterable, Iterator, Optional, Sequence, Tuple

import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader

from capymoa.base import BatchClassifier
from capymoa.ocl.events import Dispatcher, Handler
from capymoa.ocl.evaluation.events import TestTaskBegin, TrainTaskBegin
from capymoa.ocl.util._buffer_list import BufferList
from capymoa.ocl.util._optim import reset_optimizer_state
from capymoa.ocl.util._replay import SlidingWindow
from capymoa.stream._stream import Schema

NEG_INF = float("-inf")


def trainable_params(model: nn.Module) -> Iterator[Tensor]:
    """Yields the model's parameters that require gradients."""
    return (p for p in model.parameters() if p.requires_grad)


def weighted_l2_reg(
    params: Iterable[Tensor],
    anchor_params: Iterable[Tensor],
    importance: Iterable[Tensor],
    device: torch.device,
) -> Tensor:
    """Compute a MAS-style weighted L2 regularisation term."""
    l2 = torch.tensor(0.0, device=device)
    for param, anchor_param, param_importance in zip(
        params, anchor_params, importance, strict=True
    ):
        assert param.shape == anchor_param.shape
        l2 += (param_importance * (param - anchor_param) ** 2).sum()
    return 0.5 * l2


@torch.enable_grad()
def compute_importance(
    model: nn.Module,
    forward_fn: Callable[[Tensor], Tensor],
    dataloader: DataLoader[Tuple[Tensor, Tensor]],
    device: torch.device,
) -> Sequence[Tensor]:
    r"""Estimate MAS parameter importance from the given data loader.

    ..  math::

        \Omega_i = \frac{1}{N} \sum_{k=1}^N \left\| \frac{\partial \left( \|F(x_k)\|_2^2
        \right)}
             {\partial \theta_i}
        \right\|

    where :math:`F` is the model's forward function, :math:`x_k` is the input of the
    :math:`k`-th sample, and :math:`\theta_i` is the :math:`i`-th parameter.
    """
    model = model.train().to(device)
    importance = [torch.zeros_like(param) for param in trainable_params(model)]

    for x, _ in dataloader:
        x = x.to(device)
        model.zero_grad()

        # MAS importance uses gradients of the squared output norm.
        outputs = forward_fn(x)
        loss = outputs.norm(p=2, dim=1).pow(2).mean()
        loss.backward()

        # Accumulate absolute gradients
        for imp, param in zip(importance, trainable_params(model), strict=True):
            assert param.grad is not None
            imp.add_(param.grad.data.abs())

    # Average over the number of batches
    for imp in importance:
        imp.div_(len(dataloader))
    return importance


@torch.no_grad()
def update_importance_(
    importance: Iterable[Tensor], new_importance: Iterable[Tensor], alpha: float
) -> None:
    """In-place update of the MAS importance buffers via exponential moving average."""
    for imp, new_imp in zip(importance, new_importance, strict=True):
        imp.mul_(alpha).add_(new_imp, alpha=1 - alpha)


class MAS(BatchClassifier, nn.Module, Handler):
    """Memory Aware Synapses learner.

    Memory Aware Synapses (MAS) is a regularisation-based continual learning strategy
    that estimates per-parameter importance from the sensitivity of the model's output
    to small parameter perturbations, then penalises changes to parameters that were
    important for previous tasks [#f1]_.

    Unlike EWC and SI, MAS estimates importance from the squared L2 norm of the model's
    output rather than the task loss, so importance can be estimated without labels.
    We use a replay buffer to approximate the active task distribution when estimating
    importance.

    Alternative implementations:

    * `Original <https://github.com/rahafaljundi/MAS-Memory-Aware-Synapses>`__
    * `FACIL <https://github.com/mmasana/FACIL/blob/e09d2c83320a1aa945a6157d4875437515824dc9/src/approach/mas.py>`__
    * `Avalanche Lib <https://github.com/ContinualAI/avalanche/blob/eb075be393e1f458b2c352514ff6c17b5a2c0f4e/avalanche/training/plugins/mas.py>`__

    ..  [#f1] Aljundi, R., Babiloni, F., Elhoseiny, M., Rohrbach, M., & Tuytelaars, T.
        (2018). Memory Aware Synapses: Learning What (Not) to Forget. In V. Ferrari, M.
        Hebert, C. Sminchisescu, & Y. Weiss (Eds.), Computer Vision – ECCV 2018 (pp.
        144-161). Springer International Publishing.
        https://doi.org/10.1007/978-3-030-01219-9_9
    """

    def __init__(
        self,
        schema: Schema,
        model: torch.nn.Module,
        optimiser: torch.optim.Optimizer,
        lambda_: float,
        alpha: float = 0.5,
        buffer_capacity: int = 256,
        importance_batch_size: int = 32,
        device: torch.device = torch.device("cpu"),
        mask_test: bool = False,
        mask_train: bool = False,
        task_mask: Optional[Tensor] = None,
    ) -> None:
        """Construct a MAS learner.

        :param schema: Stream schema used by the classifier interface.
        :param model: Torch model that outputs class logits.
        :param optimiser: Optimiser used to update ``model`` parameters.
        :param lambda_: Weight of the MAS regularisation term.
        :param alpha: EMA factor weighting the *old* importance estimate (``alpha=1.0``
            means the importance never updates past its initial zero value;
            ``alpha=0.0`` keeps only the most recent estimate). RWalk's ``alpha``
            weights the new estimate instead.
        :param buffer_capacity: Replay window size used to estimate importance.
        :param importance_batch_size: Mini-batch size used when estimating importance.
        :param device: Compute device.
        :param mask_test: Whether to apply per-task masking during testing. This is a
            task incremental scenario.
        :param mask_train: Whether to apply per-task masking during training. This is
            also known as the labels trick.
        :param task_mask: Optional per-task mask applied to output logits.
        :raises ValueError: If ``lambda_`` is negative, ``alpha`` is outside ``[0, 1]``,
            or task-specific masking is requested without ``task_mask``.
        """
        super().__init__(schema, 0)
        nn.Module.__init__(self)
        if (mask_train or mask_test) and task_mask is None:
            raise ValueError(
                "Task schedule must be provided for task incremental or labels trick scenarios."
            )
        if lambda_ < 0:
            raise ValueError("lambda_ must be non-negative.")
        if not (0.0 <= alpha <= 1.0):
            raise ValueError("alpha must be in [0, 1].")

        self.device = device

        # Hyperparameters
        self._lambda = lambda_
        self._alpha = alpha
        self._importance_batch_size = importance_batch_size
        self._mask_train = mask_train
        self._mask_test = mask_test

        # Modules
        self._optimiser = optimiser
        self._model = model
        self._criterion = torch.nn.CrossEntropyLoss()
        self._buffer = SlidingWindow(buffer_capacity, schema.get_num_attributes())

        # Buffers used by MAS regularisation
        self._anchor_params = BufferList(
            [param.clone().detach() for param in trainable_params(model)]
        )
        self._importance = BufferList(
            [torch.zeros_like(param) for param in trainable_params(model)]
        )

        # Task tracking
        self._train_task = 0
        self._test_task = 0
        if task_mask is None:
            self._task_mask = None
        else:
            self._task_mask = nn.Buffer(task_mask)

        # Move all model parameters and buffers to the specified device
        self.to(device)

    def batch_train(self, x: Tensor, y: Tensor) -> None:
        self._buffer.update(x, y)
        self._model.train()
        self._optimiser.zero_grad()
        y_hat = self._train_forward(x)
        loss = self._criterion(y_hat, y)
        total_loss = loss + self._lambda * self._regularisation_loss()
        total_loss.backward()
        self._optimiser.step()

    @torch.no_grad()
    def batch_predict_proba(self, x: Tensor) -> Tensor:
        self._model.eval()
        y_hat = self._test_forward(x)
        return torch.softmax(y_hat, dim=1)

    def attach_with(self, source: Dispatcher) -> "MAS":
        source.subscribe(TrainTaskBegin, self._on_train_task_begin)
        source.subscribe(TestTaskBegin, self._on_test_task_begin)
        return self

    def _on_train_task_begin(self, event: TrainTaskBegin) -> None:
        reset_optimizer_state(self._optimiser)
        if event.train_task > 0:
            # Estimate importance before advancing _train_task: _importance_forward
            # reads _train_task to pick the mask, and must use the outgoing task's.
            self._update_importance()
            self._update_anchor_params()
        self._train_task = event.train_task

    def _on_test_task_begin(self, event: TestTaskBegin) -> None:
        self._test_task = event.test_task

    def _update_importance(self) -> None:
        """Estimate and accumulate MAS importance from the replay buffer."""
        dataset = self._buffer.dataset_view()
        dataloader = DataLoader(
            dataset, batch_size=self._importance_batch_size, shuffle=False
        )
        task_importance = compute_importance(
            self._model,
            self._importance_forward,
            dataloader,  # type: ignore[arg-type]
            self.device,
        )
        update_importance_(self._importance, task_importance, self._alpha)

    def _update_anchor_params(self) -> None:
        """Update anchored parameters to the current model weights."""
        for param, anchor_param in zip(
            trainable_params(self._model), self._anchor_params, strict=True
        ):
            anchor_param.copy_(param.detach())

    def _test_forward(self, x: Tensor) -> Tensor:
        """Compute logits for inference, optionally applying a test-task mask."""
        y_hat = self._model(x)
        if self._task_mask is not None and self._mask_test:
            y_hat = y_hat.masked_fill(self._task_mask[self._test_task] == 0, NEG_INF)
        return y_hat

    def _train_forward(self, x: Tensor) -> Tensor:
        """Compute logits for training, optionally applying a train-task mask."""
        y_hat = self._model(x)
        if self._task_mask is not None and self._mask_train:
            y_hat = y_hat.masked_fill(self._task_mask[self._train_task] == 0, NEG_INF)
        return y_hat

    def _importance_forward(self, x: Tensor) -> Tensor:
        """Compute logits used to estimate importance.

        Masked-out logits are omitted rather than set to ``-inf`` since MAS'
        importance estimate involves squaring the output, which turns infinities
        into NaNs.
        """
        y_hat = self._model(x)
        if self._task_mask is not None and self._mask_train:
            y_hat = y_hat[:, self._task_mask[self._train_task]]
        return y_hat

    def _regularisation_loss(self) -> Tensor:
        """Return the MAS regularisation loss for the current task."""
        if self._train_task < 1:
            return torch.tensor(0.0, device=self.device)

        return weighted_l2_reg(
            trainable_params(self._model),
            self._anchor_params,
            self._importance,
            device=self.device,
        )

    def __str__(self) -> str:
        return f"MAS(lambda_={self._lambda}, alpha={self._alpha})"
