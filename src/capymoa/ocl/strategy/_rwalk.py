from typing import Iterable, Iterator, Optional, Sequence

import torch
from torch import Tensor, nn
from torch.nn.functional import relu

from capymoa.base import BatchClassifier
from capymoa.ocl.events import Dispatcher, Handler
from capymoa.ocl.evaluation.events import TestTaskBegin, TrainTaskBegin
from capymoa.ocl.util._buffer_list import BufferList
from capymoa.ocl.util._optim import reset_optimizer_state
from capymoa.stream._stream import Schema

EPSILON = torch.finfo(torch.float32).eps
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
    """Compute an RWalk-style weighted L2 regularisation term."""
    l2 = torch.tensor(0.0, device=device)
    for param, anchor_param, param_importance in zip(
        params, anchor_params, importance, strict=True
    ):
        assert param.shape == anchor_param.shape
        l2 += (param_importance * (param - anchor_param) ** 2).sum()
    return 0.5 * l2


@torch.no_grad()
def update_importances_(
    importances: Sequence[Tensor], grads: Sequence[Tensor], alpha: float
) -> None:
    """In-place update of the EMA of squared unregularised gradients."""
    for importance, grad in zip(importances, grads, strict=True):
        importance.mul_(1 - alpha).add_(grad.square(), alpha=alpha)


@torch.no_grad()
def update_scores_(
    scores: Sequence[Tensor],
    params: Sequence[Tensor],
    losses: Sequence[Tensor],
    importances: Sequence[Tensor],
    old_params: Sequence[Tensor],
) -> None:
    """In-place accumulation of checkpoint scores using the RWalk denominator."""
    for score, param, loss, imp, old_param in zip(
        scores, params, losses, importances, old_params, strict=True
    ):
        assert score.shape == param.shape == loss.shape == imp.shape == old_param.shape
        score.add_(loss / (0.5 * imp * (param - old_param).square() + EPSILON))


@torch.no_grad()
def update_task_scores_(scores: Sequence[Tensor], old_scores: Sequence[Tensor]) -> None:
    """In-place blend of task scores with the scores from previous tasks."""
    for score, old_score in zip(scores, old_scores, strict=True):
        score.add_(old_score).mul_(0.5)


@torch.no_grad()
def set_penalties_(
    penalties: Sequence[Tensor],
    importances: Sequence[Tensor],
    scores: Sequence[Tensor],
) -> None:
    """In-place combination of importance and positive scores into RWalk penalties."""
    max_score = max(s.max() for s in scores).clamp_min(EPSILON)
    max_importance = max(i.max() for i in importances).clamp_min(EPSILON)

    for penalty, importance, score in zip(penalties, importances, scores, strict=True):
        penalty.copy_(importance / max_importance + relu(score) / max_score)


@torch.no_grad()
def copy_params_(module: nn.Module, dst: Sequence[Tensor]) -> None:
    """Copy trainable parameters into a pre-allocated list."""
    for param, dst_tensor in zip(trainable_params(module), dst, strict=True):
        dst_tensor.copy_(param.detach())


@torch.no_grad()
def copy_grads_(module: nn.Module, dst: Sequence[Tensor]) -> None:
    """Copy trainable parameters' gradients into a pre-allocated list."""
    for param, dst_tensor in zip(trainable_params(module), dst, strict=True):
        assert param.grad is not None
        dst_tensor.copy_(param.grad.detach())


@torch.no_grad()
def accumulate_loss_(
    losses: Sequence[Tensor],
    params: Iterable[Tensor],
    old_params: Sequence[Tensor],
    grads: Sequence[Tensor],
) -> None:
    """In-place update of the first-order approximation of the loss variation."""
    for loss, param, old_param, grad in zip(
        losses, params, old_params, grads, strict=True
    ):
        loss.sub_(grad * (param.detach() - old_param))


@torch.no_grad()
def zero_(buffers: Sequence[Tensor]) -> None:
    """In-place zeroing of a sequence of tensors."""
    for buffer in buffers:
        buffer.zero_()


class RWalk(BatchClassifier, nn.Module, Handler):
    """Riemannian Walk (RWalk) learner.

    RWalk [#f1]_ is a regularisation-based continual learning strategy that, like EWC,
    augments the task loss with a weighted quadratic penalty on parameter changes. The
    penalty weights combine an exponential moving average of squared gradients with
    trajectory scores that estimate how sensitive the loss is to parameter updates,
    accumulated online between periodic checkpoints.

    Alternative implementations:

    * `Original (as part of A-GEM) <https://github.com/facebookresearch/agem/tree/main>`__
    * `FACIL <https://github.com/mmasana/FACIL/blob/e09d2c83320a1aa945a6157d4875437515824dc9/src/approach/r_walk.py>`__
    * `Avalanche Lib <https://github.com/ContinualAI/avalanche/blob/eb075be393e1f458b2c352514ff6c17b5a2c0f4e/avalanche/training/plugins/rwalk.py>`__

    ..  [#f1] Chaudhry, A., Dokania, P. K., Ajanthan, T., & Torr, P. H. S. (2018).
        Riemannian Walk for Incremental Learning: Understanding Forgetting and
        Intransigence. In V. Ferrari, M. Hebert, C. Sminchisescu, & Y. Weiss (Eds.),
        Computer Vision – ECCV 2018 (pp. 556-572). Springer International Publishing.
        https://doi.org/10.1007/978-3-030-01252-6_33
    """

    def __init__(
        self,
        schema: Schema,
        model: torch.nn.Module,
        optimiser: torch.optim.Optimizer,
        lambda_: float,
        alpha: float = 0.9,
        delta_t: int = 10,
        device: torch.device = torch.device("cpu"),
        mask_test: bool = False,
        mask_train: bool = False,
        task_mask: Optional[Tensor] = None,
    ) -> None:
        """Construct an RWalk learner.

        :param schema: Stream schema used by the classifier interface.
        :param model: Torch model that outputs class logits.
        :param optimiser: Optimiser used to update ``model`` parameters.
        :param lambda_: Weight of the RWalk regularisation term.
        :param alpha: EMA decay factor weighting the *new* gradient estimate
            (``alpha=1.0`` keeps only the most recent estimate). MAS's ``alpha``
            weights the old estimate instead.
        :param delta_t: Number of training steps between score checkpoints.
        :param device: Compute device.
        :param mask_test: Whether to apply per-task masking during testing. This is a
            task incremental scenario.
        :param mask_train: Whether to apply per-task masking during training. This is
            also known as the labels trick.
        :param task_mask: Optional per-task mask applied to output logits.
        :raises ValueError: If ``lambda_`` is negative, ``alpha`` is outside ``[0, 1]``,
            ``delta_t`` is less than 1, or task-specific masking is requested without
            ``task_mask``.
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
        if delta_t < 1:
            raise ValueError("delta_t must be at least 1.")

        self.device = device

        # Hyperparameters
        self._lambda = lambda_
        self._alpha = alpha
        self._delta_t = delta_t
        self._mask_train = mask_train
        self._mask_test = mask_test

        # Modules
        self._optimiser = optimiser
        self._model = model
        self._criterion = torch.nn.CrossEntropyLoss()

        # Persistent regularisation buffers
        params = list(trainable_params(model))
        self._anchor_params = BufferList([param.clone().detach() for param in params])
        self._penalties = BufferList([torch.zeros_like(param) for param in params])
        self._task_scores = BufferList([torch.zeros_like(param) for param in params])

        # Task-local running statistics
        self._iter_importances = BufferList([torch.zeros_like(p) for p in params])
        self._iter_grads = BufferList([torch.zeros_like(param) for param in params])
        self._pre_step_params = BufferList([torch.zeros_like(p) for p in params])
        self._checkpoint_params = BufferList([p.clone().detach() for p in params])
        self._checkpoint_losses = BufferList([torch.zeros_like(p) for p in params])
        self._checkpoint_scores = BufferList([torch.zeros_like(p) for p in params])

        # Task tracking
        self._train_task = 0
        self._test_task = 0
        self._steps_since_checkpoint = 0
        self._has_completed_task = False
        if task_mask is None:
            self._task_mask = None
        else:
            self._task_mask = nn.Buffer(task_mask)

        # Move all model parameters and buffers to the specified device
        self.to(device)

    def batch_train(self, x: Tensor, y: Tensor) -> None:
        self._model.train()

        self._optimiser.zero_grad()
        y_hat = self._train_forward(x)
        loss = self._criterion(y_hat, y)
        loss.backward()

        copy_params_(self._model, self._pre_step_params)
        copy_grads_(self._model, self._iter_grads)
        update_importances_(self._iter_importances, self._iter_grads, self._alpha)

        if self._train_task > 0:
            reg_loss = self._lambda * weighted_l2_reg(
                trainable_params(self._model),
                self._anchor_params,
                self._penalties,
                device=self.device,
            )
            reg_loss.backward()

        self._optimiser.step()

        accumulate_loss_(
            self._checkpoint_losses,
            trainable_params(self._model),
            self._pre_step_params,
            self._iter_grads,
        )

        self._steps_since_checkpoint += 1
        if self._steps_since_checkpoint >= self._delta_t:
            self._flush_checkpoint_scores()

    @torch.no_grad()
    def batch_predict_proba(self, x: Tensor) -> Tensor:
        self._model.eval()
        y_hat = self._test_forward(x)
        return torch.softmax(y_hat, dim=1)

    def attach_with(self, source: Dispatcher) -> "RWalk":
        source.subscribe(TrainTaskBegin, self._on_train_task_begin)
        source.subscribe(TestTaskBegin, self._on_test_task_begin)
        return self

    def _on_train_task_begin(self, event: TrainTaskBegin) -> None:
        reset_optimizer_state(self._optimiser)
        if event.train_task > 0:
            self._finalise_previous_task()
        self._train_task = event.train_task

    def _on_test_task_begin(self, event: TestTaskBegin) -> None:
        self._test_task = event.test_task

    @torch.no_grad()
    def _finalise_previous_task(self) -> None:
        """Turn the just-finished task statistics into penalties for the next task."""
        if self._steps_since_checkpoint > 0:
            self._flush_checkpoint_scores()

        if self._has_completed_task:
            update_task_scores_(self._checkpoint_scores, self._task_scores)

        for task_score, checkpoint_score in zip(
            self._task_scores, self._checkpoint_scores, strict=True
        ):
            task_score.copy_(checkpoint_score)

        set_penalties_(self._penalties, self._iter_importances, self._task_scores)
        copy_params_(self._model, self._anchor_params)

        zero_(self._checkpoint_losses)
        zero_(self._checkpoint_scores)
        zero_(self._iter_importances)
        copy_params_(self._model, self._checkpoint_params)
        self._steps_since_checkpoint = 0
        self._has_completed_task = True

    @torch.no_grad()
    def _flush_checkpoint_scores(self) -> None:
        """Commit the current checkpoint segment into the task scores."""
        update_scores_(
            scores=self._checkpoint_scores,
            params=list(trainable_params(self._model)),
            losses=self._checkpoint_losses,
            importances=self._iter_importances,
            old_params=self._checkpoint_params,
        )
        zero_(self._checkpoint_losses)
        copy_params_(self._model, self._checkpoint_params)
        self._steps_since_checkpoint = 0

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

    def __str__(self) -> str:
        return f"RWalk(lambda_={self._lambda}, alpha={self._alpha}, delta_t={self._delta_t})"
