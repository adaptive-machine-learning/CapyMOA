# MIT License
#
# Copyright (c) 2023 Albin Soutif
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""Layerwise Proximal Replay with experience replay."""

from collections.abc import Callable
from math import isfinite

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from capymoa.base import BatchClassifier
from capymoa.core import Instance, LabelProbabilities
from capymoa.ocl.util._replay import ReservoirSampler
from capymoa.stream import Schema


def _covariance(x: Tensor, layer: nn.Module) -> tuple[Tensor, int]:
    """Return uncentered input sums and activations per example, including bias."""
    if isinstance(layer, nn.Conv2d):
        patches = F.unfold(
            x, layer.kernel_size, layer.dilation, layer.padding, layer.stride
        )
        per_example = patches.shape[2]
        x = patches.transpose(1, 2).reshape(-1, patches.shape[1])
    elif isinstance(layer, nn.BatchNorm2d):
        per_example = x.shape[-2] * x.shape[-1]
        x = x.transpose(0, 1).flatten(1)
        if layer.track_running_stats:
            mean, variance = layer.running_mean[:, None], layer.running_var[:, None]
        else:
            # Keep the reference's global statistics and variance normalization.
            mean, variance = x.mean(), x.var(unbiased=False)
        x = (x - mean) / (variance + layer.eps)
        x = x.unsqueeze(-1)
        if layer.bias is not None:
            x = torch.cat((x, torch.ones_like(x)), dim=-1)
        return x.transpose(1, 2) @ x, per_example
    else:
        if x.ndim != 2:
            raise ValueError("LPR requires two-dimensional inputs to Linear layers")
        per_example = 1
    if layer.bias is not None:
        x = torch.cat((x, x.new_ones(len(x), 1)), dim=1)
    return x.T @ x, per_example


@torch.no_grad()
def _precondition_gradient(layer: nn.Module, inverse: Tensor) -> None:
    """Right-multiply a joint weight/bias gradient by its cached inverse."""
    if layer.weight.grad is None:
        return  # The layer may belong to an inactive branch.
    if layer.bias is not None and layer.bias.grad is None:
        return
    gradient = layer.weight.grad.reshape(layer.weight.shape[0], -1)
    if layer.bias is not None:
        gradient = torch.cat((gradient, layer.bias.grad.reshape(-1, 1)), dim=1)
    transformed = (
        torch.einsum("ci,cij->cj", gradient, inverse)
        if inverse.ndim == 3
        else gradient @ inverse
    )
    if layer.bias is None:
        layer.weight.grad.copy_(transformed.reshape_as(layer.weight))
    else:
        layer.weight.grad.copy_(transformed[:, :-1].reshape_as(layer.weight))
        layer.bias.grad.copy_(transformed[:, -1].reshape_as(layer.bias))


class LPR(BatchClassifier):
    """Layerwise Proximal Replay (LPR).

    LPR [#lpr]_ preconditions mean cross-entropy gradients on current and replay
    examples with cached inverses of ``I + omega * Z.T @ Z / n``. ``Z`` holds
    historical layer inputs, including bias, and ``n`` counts memory examples.

    Supports float32/float64 Linear, ungrouped zero-padded Conv2d, and affine
    BatchNorm2d layers. Linear inputs must be matrices; supported layers may run
    once per forward pass. Other layers keep their usual gradients. Use SGD
    without momentum or weight decay for the paper's update rule.

    Inverses refresh after the training forward pass, before backward, using
    unaugmented memory sampled without replacement in evaluation mode. Spatial
    layers scale covariance by ``activations_per_example ** -beta``. Current
    examples enter the reservoir after all substeps.

    BatchNorm covariance follows the reference code: divide by ``variance + eps``
    without a square root, using global statistics if running stats are disabled.
    Private sampling means matching seeds need not match reference draws.

    >>> from capymoa.ocl.strategy import LPR
    >>> schema = Schema.from_custom(
    ...     ["x0", "x1", "class"], "class", {"class": ["a", "b"]}
    ... )
    >>> model = nn.Linear(2, 2)
    >>> learner = LPR(schema, model, torch.optim.SGD(model.parameters(), lr=0.1))
    >>> learner.batch_train(torch.eye(2), torch.tensor([0, 1]))
    >>> learner.batch_train(torch.ones(2, 2), torch.tensor([1, 0]))
    >>> learner.batch_predict_proba(torch.ones(3, 2)).shape
    torch.Size([3, 2])

    .. [#lpr] Yoo, Jinsoo, Yunpeng Liu, Frank Wood, and Geoff Pleiss.
        "Layerwise Proximal Replay: A Proximal Point Method for Online
        Continual Learning." ICML, PMLR 235, 2024, pp. 57199-57216.
        https://proceedings.mlr.press/v235/yoo24a.html
    """

    def __init__(
        self,
        schema: Schema,
        model: nn.Module,
        optimiser: torch.optim.Optimizer,
        augment: Callable[[Tensor], Tensor] | None = None,
        *,
        omega_0: float = 1.0,
        beta: float = 1.0,
        linear_omega_0: float | None = None,
        conv_omega_0: float | None = None,
        bn_omega_0: float | None = None,
        conv_beta: float | None = None,
        bn_beta: float | None = None,
        buffer_capacity: int = 200,
        replay_batch_size: int = 10,
        substeps: int = 1,
        update_every: int | None = None,
        update_n_data: int | None = None,
        update_batch_size: int = 100,
        seed: int = 0,
        device: torch.device | str = "cpu",
    ) -> None:
        """Initialize LPR.

        :param schema: Classification schema and model input shape.
        :param model: Classifier returning one logit per class.
        :param optimiser: Optimizer for this model, receiving transformed gradients.
        :param augment: Optional shape-preserving augmentation; memory stays raw.
        :param omega_0: Non-negative stability strength; zero gives ER without overrides.
        :param beta: Non-negative exponent for spatial scaling.
        :param linear_omega_0: Optional stability strength for Linear layers.
        :param conv_omega_0: Optional stability strength for Conv2d layers.
        :param bn_omega_0: Optional stability strength for BatchNorm2d layers.
        :param conv_beta: Optional spatial exponent for Conv2d layers.
        :param bn_beta: Optional spatial exponent for BatchNorm2d layers.
        :param buffer_capacity: Maximum stored examples.
        :param replay_batch_size: Maximum replay examples per step, drawn without replacement.
        :param substeps: Optimizer steps per batch, each with a new replay draw.
        :param update_every: Refresh interval in optimizer steps; None uses ``substeps``.
        :param update_n_data: Exact maximum memory examples per refresh; None uses all.
        :param update_batch_size: Memory chunk size for covariance computation.
        :param seed: Private buffer and replay sampling seed.
        :param device: Device for the model, memory and cached inverses.
        """
        super().__init__(schema, seed)
        if not schema.is_classification():
            raise ValueError("LPR requires a classification schema")
        for name, value in (
            ("buffer_capacity", buffer_capacity),
            ("replay_batch_size", replay_batch_size),
            ("substeps", substeps),
            ("update_batch_size", update_batch_size),
            ("update_every", update_every),
            ("update_n_data", update_n_data),
        ):
            if value is None and name in {"update_every", "update_n_data"}:
                continue
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        for name, value in (
            ("omega_0", omega_0),
            ("beta", beta),
            ("linear_omega_0", linear_omega_0),
            ("conv_omega_0", conv_omega_0),
            ("bn_omega_0", bn_omega_0),
            ("conv_beta", conv_beta),
            ("bn_beta", bn_beta),
        ):
            if value is None and name not in {"omega_0", "beta"}:
                continue
            if not isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and non-negative")
        if augment is not None and not callable(augment):
            raise TypeError("augment must be callable or None")
        if not isinstance(optimiser, torch.optim.Optimizer):
            raise TypeError("optimiser must be a torch optimizer")
        self.device = torch.device(device)
        self._model = model.to(self.device)
        self.x_dtype = next(model.parameters(), torch.empty(0)).dtype
        if self.x_dtype not in (torch.float32, torch.float64):
            raise ValueError("LPR supports float32 and float64 models")
        owned = {id(p) for g in optimiser.param_groups for p in g["params"]}
        model_params = {id(p) for p in model.parameters()}
        if (
            not {id(p) for p in model.parameters() if p.requires_grad} <= owned
            or not owned <= model_params
        ):
            raise ValueError("Optimizer must cover all trainable model parameters only")
        self._layers = {}
        for name, layer in model.named_modules():
            if not isinstance(layer, (nn.Linear, nn.Conv2d, nn.BatchNorm2d)):
                continue
            if layer.weight is None or not any(
                p.requires_grad for p in layer.parameters(recurse=False)
            ):
                continue
            if not layer.weight.requires_grad or (
                layer.bias is not None and not layer.bias.requires_grad
            ):
                raise ValueError(
                    "LPR requires weight and bias to be trainable together"
                )
            if isinstance(layer, nn.Conv2d) and (
                layer.groups != 1
                or layer.padding_mode != "zeros"
                or isinstance(layer.padding, str)
            ):
                raise ValueError(
                    "LPR requires ungrouped Conv2d with numeric zero padding"
                )
            self._layers[name] = layer
        if not self._layers:
            raise ValueError(
                "LPR requires a trainable Linear, Conv2d or BatchNorm2d layer"
            )
        self._optimiser = optimiser
        self._augment = (
            augment.to(self.device) if isinstance(augment, nn.Module) else augment
        )
        self._shape = tuple(schema.shape)
        self._num_features = schema.get_num_attributes()
        self._rng = torch.Generator().manual_seed(seed)
        self._buffer = ReservoirSampler(
            buffer_capacity,
            {"x": ((self._num_features,), self.x_dtype), "y": ((), self.y_dtype)},
            self._rng,
        ).to(self.device)
        self._strengths = (omega_0, linear_omega_0, conv_omega_0, bn_omega_0)
        self._exponents = (beta, conv_beta, bn_beta)
        self._replay_batch_size = replay_batch_size
        self._substeps = substeps
        self._update_every = substeps if update_every is None else update_every
        self._update_n_data = update_n_data
        self._update_batch_size = update_batch_size
        self._iterations = 0
        self._preconditioners: dict[str, Tensor] = {}
        self._preconditioner_updates = 0
        self.last_loss: float | None = None

    def _prepare_x(self, x: Tensor) -> Tensor:
        """Validate and flatten a batch for storage."""
        if x.ndim < 2 or len(x) == 0:
            raise ValueError("x must contain a non-empty batch")
        x = x.reshape(len(x), -1)
        if x.shape[1] != self._num_features:
            raise ValueError("feature count does not match the schema")
        return x.to(device=self.device, dtype=self.x_dtype)

    def _sample(self, count: int) -> dict[str, Tensor]:
        """Sample memory without replacement."""
        indices = torch.randperm(self._buffer.count, generator=self._rng)[:count].to(
            self.device
        )
        return {key: value[indices] for key, value in self._buffer.array().items()}

    def _scale(self, layer: nn.Module, per_example: int) -> float:
        """Return the layer's stability and spatial scale."""
        omega, linear, conv, bn = self._strengths
        beta, conv_beta, bn_beta = self._exponents
        if isinstance(layer, nn.Linear):
            return omega if linear is None else linear
        if isinstance(layer, nn.Conv2d):
            strength, exponent = conv, conv_beta
        else:
            strength, exponent = bn, bn_beta
        return (omega if strength is None else strength) / per_example ** (
            beta if exponent is None else exponent
        )

    @torch.no_grad()
    def _refresh_preconditioners(self) -> None:
        """Cache covariance inverses from historical layer inputs."""
        n = (
            self._buffer.count
            if self._update_n_data is None
            else min(self._buffer.count, self._update_n_data)
        )
        if n == 0:
            return
        memory = self._sample(n)["x"]
        inputs, totals, counts = {}, {}, {}

        def capture(name):
            def hook(layer, args, output):
                if name in inputs:
                    raise ValueError(
                        "LPR requires each supported layer to be used once per forward"
                    )
                inputs[name] = args[0].detach().clone()

            return hook

        hooks = []
        modes = {layer: layer.training for layer in self._model.modules()}
        try:
            for name, layer in self._layers.items():
                hooks.append(layer.register_forward_hook(capture(name)))
            self._model.eval()
            for batch in memory.split(self._update_batch_size):
                inputs.clear()
                self._model(batch.reshape(-1, *self._shape))
                for name, values in inputs.items():
                    covariance, per_example = _covariance(values, self._layers[name])
                    scaled = self._scale(self._layers[name], per_example) * covariance
                    totals[name] = totals.get(name, 0) + scaled
                    counts[name] = counts.get(name, 0) + len(batch)
        finally:
            for hook in hooks:
                hook.remove()
            for layer, training in modes.items():
                layer.training = training
        inverses = {}
        for name, total in totals.items():
            covariance = total / counts[name]
            identity = torch.eye(
                covariance.shape[-1], device=self.device, dtype=covariance.dtype
            )
            inverses[name] = torch.linalg.inv(covariance + identity)
        self._preconditioners = inverses
        self._preconditioner_updates += 1

    def batch_train(self, x: Tensor, y: Tensor) -> None:
        """Train on current and replay data with preconditioned gradients."""
        x = self._prepare_x(x)
        y = y.to(device=self.device, dtype=self.y_dtype)
        if (
            y.shape != (len(x),)
            or ((y < 0) | (y >= self.schema.get_num_classes())).any()
        ):
            raise ValueError("y must contain one valid class index per example")
        losses = []
        self._model.train()
        for _ in range(self._substeps):
            train_x, train_y = x.reshape(-1, *self._shape), y
            if self._buffer.count:
                replay = self._sample(self._replay_batch_size)
                train_x = torch.cat((train_x, replay["x"].reshape(-1, *self._shape)))
                train_y = torch.cat((y, replay["y"]))
            if self._augment is not None:
                augmented = self._augment(train_x.clone())
                if augmented.shape != train_x.shape:
                    raise ValueError("augment must preserve the batch shape")
                train_x = augmented
            self._optimiser.zero_grad()
            loss = F.cross_entropy(self._model(train_x), train_y)
            if self._buffer.count and self._iterations % self._update_every == 0:
                self._refresh_preconditioners()
            loss.backward()
            for name, inverse in self._preconditioners.items():
                _precondition_gradient(self._layers[name], inverse)
            self._optimiser.step()
            self._iterations += 1
            losses.append(loss.detach())
        self._buffer.update(x=x, y=y)
        self.last_loss = torch.stack(losses).mean().item()

    @torch.no_grad()
    def batch_predict_proba(self, x: Tensor) -> Tensor:
        """Predict probabilities and restore model modes."""
        modes = {layer: layer.training for layer in self._model.modules()}
        self._model.eval()
        try:
            return self._model(self._prepare_x(x).reshape(-1, *self._shape)).softmax(1)
        finally:
            for layer, training in modes.items():
                layer.training = training

    def predict_proba(self, instance: Instance) -> LabelProbabilities:
        """Return one instance's probabilities as a CPU array."""
        x = torch.as_tensor(instance.x).reshape(1, -1)
        return (
            self.batch_predict_proba(x)
            .to(device="cpu", dtype=torch.float64)
            .numpy()
            .reshape(-1)
        )

    def __str__(self) -> str:
        return f"LPR(omega_0={self._strengths[0]}, beta={self._exponents[0]}, buffer_capacity={self._buffer.capacity})"
