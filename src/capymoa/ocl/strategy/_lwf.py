from copy import deepcopy
from typing import Optional

import torch
from torch import Tensor, nn

from capymoa.base import BatchClassifier
from capymoa.ocl.events import Dispatcher, Handler
from capymoa.ocl.evaluation.events import TrainTaskBegin
from capymoa.ocl.util._optim import reset_optimizer_state
from capymoa.ocl.util.functional import hinton_distillation_loss
from capymoa.stream._stream import Schema


class LWF(BatchClassifier, nn.Module, Handler):
    """Learning Without Forgetting (LwF).

    LwF [#f1]_ is a regularisation-based continual learning strategy that distils
    predictions from a frozen teacher snapshot of the previous task while learning the
    current task.

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
    ) -> None:
        """Construct an LWF learner.

        :param schema: Stream schema used by the classifier interface.
        :param model: Torch model that outputs class logits.
        :param optimiser: Optimiser used to update ``model`` parameters.
        :param alpha: Weight of the distillation loss term.
        :param temperature: Distillation temperature.
        :param device: Compute device.
        :raises ValueError: If ``alpha`` is negative or ``temperature`` is not
            positive.
        """
        super().__init__(schema, 0)
        nn.Module.__init__(self)
        if alpha < 0:
            raise ValueError("alpha must be non-negative.")
        if temperature <= 0:
            raise ValueError("temperature must be greater than zero.")

        self.device = device

        self._alpha = alpha
        self._temperature = temperature

        self._optimiser = optimiser
        self._model = model
        self._criterion = torch.nn.CrossEntropyLoss()

        self._teacher: Optional[torch.nn.Module] = None
        self._train_task = 0

        # Move all model parameters and buffers to the specified device
        self.to(device)

    def batch_train(self, x: Tensor, y: Tensor) -> None:
        self._model.train()
        self._optimiser.zero_grad()

        student_logits = self._model(x)
        task_loss = self._criterion(student_logits, y)
        total_loss = task_loss + self._alpha * self._distillation_loss(
            x, student_logits
        )

        total_loss.backward()
        self._optimiser.step()

    @torch.no_grad()
    def batch_predict_proba(self, x: Tensor) -> Tensor:
        self._model.eval()
        y_hat = self._model(x)
        return torch.softmax(y_hat, dim=1)

    def attach_with(self, source: Dispatcher) -> "LWF":
        source.subscribe(TrainTaskBegin, self._on_train_task_begin)
        return self

    def _on_train_task_begin(self, event: TrainTaskBegin) -> None:
        reset_optimizer_state(self._optimiser)
        if event.train_task > 0:
            self._teacher = (
                deepcopy(self._model).to(self.device).eval().requires_grad_(False)
            )
        self._train_task = event.train_task

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
