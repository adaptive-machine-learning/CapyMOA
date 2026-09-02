from copy import deepcopy
from typing import Optional

import torch
from torch import Tensor, nn

from capymoa.base import BatchClassifier
from capymoa.ocl.events import Dispatcher, Handler
from capymoa.ocl.evaluation.events import TestTaskBegin, TrainTaskBegin
from capymoa.ocl.util._optim import reset_optimizer_state
from capymoa.ocl.util.functional import hinton_distillation_loss
from capymoa.stream._stream import Schema

NEG_INF = float("-inf")


class LWF(BatchClassifier, nn.Module, Handler):
    """Learning Without Forgetting (LwF).

    LwF [#f1]_ is a regularisation-based continual learning strategy that distils
    predictions from a frozen teacher snapshot of the previous task while learning the
    current task.

    LWF does not support task-incremental masking; its source paper has no per-task mask.

    ..  [#f1] Li, Z., & Hoiem, D. (2016). Learning without forgetting. CoRR,
        abs/1606.09282. http://arxiv.org/abs/1606.09282
    """

    def __init__(
        self,
        schema: Schema,
        model: torch.nn.Module,
        optimiser: torch.optim.Optimizer,
        alpha: float = 1.0,
        temperature: float = 2.0,
        device: torch.device = torch.device("cpu"),
        mask_test: bool = False,
        mask_train: bool = False,
        task_mask: Optional[Tensor] = None,
    ) -> None:
        """Construct an LWF learner.

        :param schema: Stream schema used by the classifier interface.
        :param model: Torch model that outputs class logits.
        :param optimiser: Optimiser used to update ``model`` parameters.
        :param alpha: Weight of the distillation loss term.
        :param temperature: Distillation temperature.
        :param device: Compute device.
        :param mask_test: Whether to apply per-task masking during testing. This is a
            task incremental scenario.
        :param mask_train: Whether to apply per-task masking during training. This is
            also known as the labels trick.
        :param task_mask: Optional per-task mask applied to output logits.
        :raises ValueError: If ``alpha`` is negative, ``temperature`` is not positive,
            or task-specific masking is requested without ``task_mask``.
        """
        super().__init__(schema, 0)
        nn.Module.__init__(self)
        if alpha < 0:
            raise ValueError("alpha must be non-negative.")
        if temperature <= 0:
            raise ValueError("temperature must be greater than zero.")
        if (mask_train or mask_test) and task_mask is None:
            raise ValueError(
                "Task schedule must be provided for task incremental or labels trick scenarios."
            )

        self.device = device

        self._alpha = alpha
        self._temperature = temperature
        self._mask_train = mask_train
        self._mask_test = mask_test

        self._optimiser = optimiser
        self._model = model
        self._criterion = torch.nn.CrossEntropyLoss()

        self._teacher: Optional[torch.nn.Module] = None
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
        self._optimiser.zero_grad()

        raw_logits = self._model(x)
        student_logits = self._apply_train_mask(raw_logits)
        task_loss = self._criterion(student_logits, y)
        total_loss = task_loss + self._alpha * self._distillation_loss(x, raw_logits)

        total_loss.backward()
        self._optimiser.step()

    @torch.no_grad()
    def batch_predict_proba(self, x: Tensor) -> Tensor:
        self._model.eval()
        y_hat = self._apply_test_mask(self._model(x))
        return torch.softmax(y_hat, dim=1)

    def attach_with(self, source: Dispatcher) -> "LWF":
        source.subscribe(TrainTaskBegin, self._on_train_task_begin)
        source.subscribe(TestTaskBegin, self._on_test_task_begin)
        return self

    def _on_train_task_begin(self, event: TrainTaskBegin) -> None:
        reset_optimizer_state(self._optimiser)
        if event.train_task > 0:
            self._teacher = (
                deepcopy(self._model).to(self.device).eval().requires_grad_(False)
            )
        self._train_task = event.train_task

    def _on_test_task_begin(self, event: TestTaskBegin) -> None:
        self._test_task = event.test_task

    def _apply_train_mask(self, y_hat: Tensor) -> Tensor:
        """Apply the train-task mask to logits, if enabled."""
        if self._task_mask is not None and self._mask_train:
            y_hat = y_hat.masked_fill(self._task_mask[self._train_task] == 0, NEG_INF)
        return y_hat

    def _apply_test_mask(self, y_hat: Tensor) -> Tensor:
        """Apply the test-task mask to logits, if enabled."""
        if self._task_mask is not None and self._mask_test:
            y_hat = y_hat.masked_fill(self._task_mask[self._test_task] == 0, NEG_INF)
        return y_hat

    @torch.no_grad()
    def _teacher_forward(self, x: Tensor) -> Tensor:
        if self._teacher is None:
            raise RuntimeError("Teacher model is not available before task 1.")
        return self._teacher(x)

    def _distillation_loss(self, x: Tensor, student_logits: Tensor) -> Tensor:
        if self._teacher is None:
            return torch.tensor(0.0, device=self.device)

        teacher_logits = self._teacher_forward(x)

        return hinton_distillation_loss(
            teacher_logits=teacher_logits,
            student_logits=student_logits,
            temperature=self._temperature,
        )

    def __str__(self) -> str:
        return f"LWF(alpha={self._alpha}, temperature={self._temperature})"
