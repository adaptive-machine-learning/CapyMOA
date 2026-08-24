from typing import Iterable, Iterator, Optional, Sequence

import torch
from torch import Tensor, nn

from capymoa.base import BatchClassifier
from capymoa.ocl.events import Dispatcher, Handler
from capymoa.ocl.evaluation.events import TestTaskBegin, TrainTaskBegin
from capymoa.ocl.util._buffer_list import BufferList
from capymoa.ocl.util._optim import reset_optimizer_state
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
    """Compute an SI-style weighted L2 regularisation term."""
    l2 = torch.tensor(0.0, device=device)
    for param, anchor_param, param_importance in zip(
        params, anchor_params, importance, strict=True
    ):
        assert param.shape == anchor_param.shape
        l2 += (param_importance * (param - anchor_param) ** 2).sum()
    return 0.5 * l2


@torch.no_grad()
def update_trajectory(
    trajectory: Sequence[Tensor],
    pre_step_params: Iterable[Tensor],
    post_step_params: Iterable[Tensor],
    gradients: Iterable[Tensor],
) -> None:
    r"""Update the parameter's cumulative trajectory for Synaptic Intelligence.

    Should be called after the optimizer step, using the gradients from before that
    step. This function implements Equation 2 from [#f1]_:

    ..  math::

        \begin{aligned} \int_{t^{\mu-1}}^{t^\mu} \boldsymbol{g}(\boldsymbol{\theta}(t))
        \cdot \boldsymbol{\theta}^{\prime}(t) d t & =\sum_k \int_{t^{\mu-1}}^{t^\mu}
        g_k(\theta(t)) \theta_k^{\prime}(t) d t \\ & \equiv-\sum_k \omega_k^\mu,
        \end{aligned}

    where:

    *   :math:`\boldsymbol{g}(\boldsymbol{\theta}(t))` is the gradient of the loss with
        respect to the parameters at optimization step :math:`t`.
    *   :math:`\boldsymbol{\theta}^{\prime}(t)` is the difference between the parameters
        at optimization step :math:`t` (``post_step_params``) and the parameters at
        :math:`t-1` (``pre_step_params``).

    This function updates the trajectory :math:`\omega_k^\mu` for each parameter
    :math:`k` after each optimization step.

    :param trajectory: Sequence of tensors storing the cumulative trajectory for each
        parameter. Updated in-place.
    :param pre_step_params: Parameters before the optimizer step.
    :param post_step_params: Parameters after the optimizer step.
    :param gradients: Gradients of the loss with respect to the parameters before the
        optimizer step.
    """
    for traj, pre_param, post_param, grad in zip(
        trajectory,
        pre_step_params,
        post_step_params,
        gradients,
        strict=True,
    ):
        # The negative sign ensures we measure the *decrease* in loss.
        # Trajectory (w) = -grad * delta_theta
        step_contribution = -grad * (post_param - pre_param)
        traj.add_(step_contribution)


@torch.no_grad()
def update_importance_weights_(
    importance: Sequence[Tensor],
    trajectory: Iterable[Tensor],
    start_task_params: Iterable[Tensor],
    end_task_params: Iterable[Tensor],
    damping: float = 0.1,
) -> None:
    """In-place update of the SI importance buffers.

    Calculates the new importance matrix (Omega) at the end of a task.
    """
    for omega, traj, start_param, end_param in zip(
        importance, trajectory, start_task_params, end_task_params, strict=True
    ):
        # Importance is the accumulated trajectory normalized by the total change
        # in the parameter over the whole task (plus a damping factor for numerical
        # stability).
        param_shift_squared = (end_param - start_param).pow(2)
        task_importance = traj / (param_shift_squared + damping)

        # Accumulate importance across sequential tasks
        omega.add_(task_importance)


@torch.no_grad()
def copy_grads_(module: nn.Module, dst: Sequence[Tensor]) -> None:
    """Copy gradients from a module's parameters into a pre-allocated list."""
    for param, dst_tensor in zip(trainable_params(module), dst, strict=True):
        assert param.grad is not None
        dst_tensor.copy_(param.grad.detach())


@torch.no_grad()
def copy_params_(module: nn.Module, dst: Sequence[Tensor]) -> None:
    """Copy parameters from a module into a pre-allocated list."""
    for param, dst_tensor in zip(trainable_params(module), dst, strict=True):
        dst_tensor.copy_(param.detach())


@torch.no_grad()
def reset_trajectory_(trajectory: Sequence[Tensor]) -> None:
    """In-place zeroing of the SI trajectory buffers."""
    for traj in trajectory:
        traj.zero_()


class SI(BatchClassifier, nn.Module, Handler):
    """Synaptic Intelligence learner.

    Synaptic Intelligence (SI) is a regularisation-based continual learning strategy
    that accumulates per-parameter importance online from optimization trajectories,
    then penalises changes to parameters that were important for previous tasks [#f1]_.

    Alternative implementations:

    * `Avalanche Lib <https://github.com/ContinualAI/avalanche/blob/eb075be393e1f458b2c352514ff6c17b5a2c0f4e/avalanche/training/plugins/synaptic_intelligence.py>`__
    * `FACIL <https://github.com/mmasana/FACIL/blob/e09d2c83320a1aa945a6157d4875437515824dc9/src/approach/path_integral.py>`__

    ..  [#f1] Zenke, F., Poole, B., & Ganguli, S. (2017). Continual Learning Through
        Synaptic Intelligence. International Conference on Machine Learning, 3987–3995.
    """

    def __init__(
        self,
        schema: Schema,
        model: torch.nn.Module,
        optimiser: torch.optim.Optimizer,
        lambda_: float,
        damping: float = 0.1,
        device: torch.device = torch.device("cpu"),
        mask_test: bool = False,
        mask_train: bool = False,
        task_mask: Optional[Tensor] = None,
    ) -> None:
        """Construct an SI learner.

        :param schema: Stream schema used by the classifier interface.
        :param model: Torch model that outputs class logits.
        :param optimiser: Optimiser used to update ``model`` parameters.
        :param lambda_: Weight of the SI regularisation term.
        :param damping: Damping factor added to the denominator when calculating
            importance weights.
        :param device: Compute device.
        :param mask_test: Whether to apply per-task masking during testing. This is a
            task incremental scenario.
        :param mask_train: Whether to apply per-task masking during training. This is
            also known as the labels trick.
        :param task_mask: Optional per-task mask applied to output logits.
        :raises ValueError: If task-specific masking is requested without ``task_mask``.
        """
        super().__init__(schema, 0)
        nn.Module.__init__(self)
        if (mask_train or mask_test) and task_mask is None:
            raise ValueError(
                "Task schedule must be provided for task incremental or labels trick scenarios."
            )
        if lambda_ < 0:
            raise ValueError("lambda_ must be non-negative.")
        if damping <= 0:
            raise ValueError("damping must be positive.")

        self.device = device

        # Hyperparameters
        self._lambda = lambda_
        self._eps = damping
        self._mask_train = mask_train
        self._mask_test = mask_test

        # Modules
        self._optimiser = optimiser
        self._model = model
        self._criterion = torch.nn.CrossEntropyLoss()

        # Allocate buffers for SI regularisation
        self._buf_anchor = BufferList(
            [p.clone().detach() for p in trainable_params(model)]
        )
        self._buf_importance = BufferList(
            [torch.zeros_like(p) for p in trainable_params(model)]
        )
        self._buf_pre_step_params = BufferList(
            [torch.zeros_like(p) for p in trainable_params(model)]
        )
        self._buf_trajectory = BufferList(
            [torch.zeros_like(p) for p in trainable_params(model)]
        )
        self._buf_grads = BufferList(
            [torch.zeros_like(p) for p in trainable_params(model)]
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
        self._model.train()

        # Compute unregularised loss and gradients
        self._optimiser.zero_grad()
        y_hat = self._train_forward(x)
        loss = self._criterion(y_hat, y)
        loss.backward()

        # Capture parameters before the optimiser step
        copy_params_(self._model, self._buf_pre_step_params)

        # Save unregularised gradients needed for the path integral
        copy_grads_(self._model, self._buf_grads)

        # Add SI regularisation loss (only applies after the first task)
        if self._train_task > 0:
            reg_loss = self._lambda * weighted_l2_reg(
                trainable_params(self._model),
                self._buf_anchor,
                self._buf_importance,
                device=self.device,
            )
            reg_loss.backward()

        # Apply the optimiser step
        self._optimiser.step()

        # Update the trajectory using the unregularised gradients and parameter changes
        update_trajectory(
            trajectory=self._buf_trajectory,
            pre_step_params=self._buf_pre_step_params,
            post_step_params=trainable_params(self._model),
            gradients=self._buf_grads,
        )

    @torch.no_grad()
    def batch_predict_proba(self, x: Tensor) -> Tensor:
        self._model.eval()
        y_hat = self._test_forward(x)
        return torch.softmax(y_hat, dim=1)

    def attach_with(self, source: Dispatcher) -> "SI":
        source.subscribe(TrainTaskBegin, self._on_train_task_begin)
        source.subscribe(TestTaskBegin, self._on_test_task_begin)
        return self

    def _on_train_task_begin(self, event: TrainTaskBegin) -> None:
        reset_optimizer_state(self._optimiser)
        self._train_task = event.train_task

        if self._train_task > 0:
            # Consolidate importance weights using the trajectory from the previous task
            update_importance_weights_(
                importance=self._buf_importance,
                trajectory=self._buf_trajectory,
                start_task_params=self._buf_anchor,
                end_task_params=trainable_params(self._model),
                damping=self._eps,
            )

            # Update anchors to the current model parameters
            copy_params_(self._model, self._buf_anchor)

            # Reset trajectory for the new task
            reset_trajectory_(self._buf_trajectory)

    def _on_test_task_begin(self, event: TestTaskBegin) -> None:
        self._test_task = event.test_task

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
        return f"SI(lambda_={self._lambda}, eps={self._eps})"
