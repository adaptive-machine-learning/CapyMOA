from typing import Iterable, Iterator, Optional, Sequence, Tuple, Callable
from capymoa.stream._stream import Schema
from torch import Tensor, nn
import torch
from capymoa.base import BatchClassifier
from capymoa.ocl.events import Handler, Dispatcher
from capymoa.ocl.evaluation.events import TestTaskBegin, TrainTaskBegin
from capymoa.ocl.util._buffer_list import BufferList
from capymoa.ocl.util._replay import SlidingWindow
from torch.utils.data import DataLoader


def weighted_l2_reg(
    params: Iterable[Tensor],
    anchor_params: Iterable[Tensor],
    fisher_diagonals: Iterable[Tensor],
    device: torch.device,
) -> Tensor:
    """Compute an EWC-style weighted L2 regularisation term.

    :param params: Current model parameters.
    :param anchor_params: Reference parameters from a previous task.
    :param fisher_diagonals: Diagonal Fisher information weights.
    :param device: Device used for the accumulator tensor.
    :return: Weighted L2 penalty scaled by ``1/2``.
    """
    l2 = torch.tensor(0.0, device=device)
    for param, anchor_param, fisher_diag in zip(
        params, anchor_params, fisher_diagonals, strict=True
    ):
        assert param.shape == anchor_param.shape
        l2 += (fisher_diag * (param - anchor_param) ** 2).sum()
    return l2 / 2.0


def fd_init(model: torch.nn.Module) -> Sequence[Tensor]:
    """Initialise zero-valued Fisher diagonal tensors for a model.

    :param model: Model whose parameters define the Fisher diagonal shapes.
    :return: Zero tensors matching all model parameters.
    """
    return [torch.zeros_like(param) for param in model.parameters()]


def fd_accumulate(
    fisher_diagonals: Sequence[Tensor],
    parameters: Iterator[Tensor],
    alpha: Optional[float] = None,
) -> Sequence[Tensor]:
    """Accumulates the squared gradients into the Fisher diagonal estimates.

    :param fisher_diagonals: A sequence of tensors representing the current estimates of
        the Fisher diagonals.
    :param parameters: A sequence of model parameters whose gradients have been
        computed.
    :param alpha: Decay factor for the accumulated Fisher diagonals. A value of 1.0
        corresponds to standard EWC accumulation, while values less than 1.0 implement
        a decay as in Online EWC.
    :return: Updated sequence of tensors representing the accumulated Fisher diagonals.
    """
    for fisher_diag, param in zip(fisher_diagonals, parameters, strict=True):
        if param.grad is None:
            raise ValueError(
                "Parameter gradients must be computed before updating Fisher diagonals."
            )
        if alpha is not None:
            fisher_diag.mul_(alpha).add_(param.grad.data.pow(2), alpha=(1 - alpha))
        else:
            fisher_diag.add_(param.grad.data.pow(2))
    return fisher_diagonals


def fd_compute(
    model: torch.nn.Module,
    forward_fn: Callable[[Tensor], Tensor],
    dataloader: DataLoader[Tuple[Tensor, Tensor]],
    device: torch.device,
    criterion: torch.nn.Module,
) -> Sequence[Tensor]:
    """Compute module fisher diagonals.

    :param model: A PyTorch classifier model.
    :param dataloader: A PyTorch dataloader for a classification task, yielding batches
        of (inputs, labels).
    :param device: Compute device.
    :param criterion: The loss function to use.
    :return: A sequence of tensors representing the computed Fisher diagonals.
    """
    model = model.eval().to(device)
    criterion = criterion.eval().to(device)

    fisher_diagonals = fd_init(model)
    for inputs, labels in dataloader:
        model.zero_grad()
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = forward_fn(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        fisher_diagonals = fd_accumulate(fisher_diagonals, model.parameters())
    # Average the accumulated squared gradients over the number of samples
    fisher_diagonals = [
        fisher_diag / len(dataloader) for fisher_diag in fisher_diagonals
    ]
    return fisher_diagonals


class EWC(BatchClassifier, nn.Module, Handler):
    """Elastic Weight Consolidation learner.

    Elastic Weight Consolidation (EWC) is a regularisation-based continual learning
    strategy that mitigates catastrophic forgetting by penalising changes to important
    parameters for previous tasks [#f1]_. We incorporate Online EWC-style [#f2]_ updates
    to the Fisher diagonals, which decay the importance of previous tasks' parameters
    over time based on the ``gamma`` hyperparameter.

    Usually the EWC strategy has access to the entire active task's data when estimating
    the Fisher diagonals, but instead we use a replay buffer to approximate the active
    task distribution.

    ..  [#f1] Kirkpatrick, J., Pascanu, R., Rabinowitz, N., Veness, J., Desjardins, G.,
        Rusu, A. A., Milan, K., Quan, J., Ramalho, T., Grabska-Barwinska, A., Hassabis,
        D., Clopath, C., Kumaran, D., & Hadsell, R. (2017). Overcoming catastrophic
        forgetting in neural networks. Proceedings of the National Academy of Sciences,
        114(13), 3521–3526. https://doi.org/10.1073/pnas.1611835114

    ..  [#f2] Schwarz, J., Czarnecki, W., Luketina, J., Grabska-Barwinska, A., Teh, Y.
        W., Pascanu, R., & Hadsell, R. (2018). Progress & Compress: A scalable framework
        for continual learning. In J. G. Dy & A. Krause (Eds.), Proceedings of the 35th
        International Conference on Machine Learning, ICML 2018, Stockholmsmässan,
        Stockholm, Sweden, July 10-15, 2018 (Vol. 80, pp. 4535–4544). PMLR.
        http://proceedings.mlr.press/v80/schwarz18a.html
    """

    def __init__(
        self,
        schema: Schema,
        model: torch.nn.Module,
        optimiser: torch.optim.Optimizer,
        lambda_: float,
        fim_buffer: int = 256,
        fim_batch_size: int = 32,
        device: torch.device = torch.device("cpu"),
        mask_test: bool = False,
        mask_train: bool = False,
        gamma: float = 1.0,
        task_mask: Optional[Tensor] = None,
    ) -> None:
        """Construct an EWC learner.

        :param schema: Stream schema used by the classifier interface.
        :param model: Torch model that outputs class logits.
        :param optimiser: Optimiser used to update ``model`` parameters.
        :param lambda_: Weight of the EWC regularisation term.
        :param fim_buffer: Replay window size for Fisher estimation.
        :param fim_batch_size: Mini-batch size used when estimating Fisher diagonals.
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
        self.device = device

        # Hyperparameters
        self._lambda = lambda_
        self._gamma = gamma
        self._fd_batch_size = fim_batch_size
        self._mask_train = mask_train
        self._mask_test = mask_test

        # Modules
        self._optimiser = optimiser
        self._model = model
        self._criterion = torch.nn.CrossEntropyLoss()
        self._buffer = SlidingWindow(fim_buffer, schema.get_num_attributes())

        # Buffers for anchoring the model
        self._anchor_params = BufferList(
            [param.clone().detach() for param in model.parameters()]
        )
        self._fisher_diags = BufferList(
            [torch.zeros_like(param) for param in model.parameters()]
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

    def attach_with(self, source: Dispatcher) -> "EWC":
        source.subscribe(TrainTaskBegin, self.on_train_task)
        source.subscribe(TestTaskBegin, self.on_test_task)
        return self

    def on_train_task(self, event: TrainTaskBegin) -> None:
        if event.train_task > 0:
            self._update_fisher_diags()
            self._update_anchor_params()
        self._train_task = event.train_task

    def on_test_task(self, event: TestTaskBegin) -> None:
        self._test_task = event.test_task

    def _update_fisher_diags(self) -> None:
        """Estimate and accumulate Fisher diagonals from the replay buffer."""
        dataset = self._buffer.dataset_view()
        dataloader = DataLoader(dataset, batch_size=self._fd_batch_size, shuffle=False)
        task_fisher_diags = fd_compute(
            self._model,
            self._train_forward,
            dataloader,  # type: ignore
            self.device,
            self._criterion,
        )
        # Update the fisher diagonals buffer with the computed values
        for i in range(len(self._fisher_diags)):
            self._fisher_diags[i].mul_(self._gamma).add_(task_fisher_diags[i])

    def _update_anchor_params(self) -> None:
        """Update anchored parameters to the current model weights."""
        for param, anchor_param in zip(
            self._model.parameters(), self._anchor_params, strict=True
        ):
            anchor_param.copy_(param.detach())

    def _test_forward(self, x: Tensor) -> Tensor:
        """Compute logits for inference, optionally applying a test-task mask."""
        y_hat = self._model(x)
        if self._task_mask is not None and self._mask_test:
            y_hat = self._task_mask[self._test_task] * y_hat
        return y_hat

    def _train_forward(self, x: Tensor) -> Tensor:
        """Compute logits for training, optionally applying a train-task mask."""
        y_hat = self._model(x)
        if self._task_mask is not None and self._mask_train:
            y_hat = self._task_mask[self._train_task] * y_hat
        return y_hat

    def _regularisation_loss(self) -> torch.Tensor:
        """Return the EWC regularisation loss for the current task."""
        if self._train_task < 1:
            return torch.tensor(0.0, device=self.device)
        return weighted_l2_reg(
            self._model.parameters(),
            self._anchor_params,
            self._fisher_diags,
            device=self.device,
        )
