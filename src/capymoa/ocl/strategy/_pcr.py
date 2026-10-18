"""Proxy-based Contrastive Replay."""

from collections.abc import Callable, Iterator
from math import isfinite

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch.optim import Optimizer

from capymoa.base import BatchClassifier
from capymoa.core import Instance, LabelProbabilities
from capymoa.ocl.util._replay import ReservoirSampler
from capymoa.stream import Schema


class _ProxyContrastiveLoss(nn.Module):
    """Contrast features with one class proxy per example, keeping duplicates."""

    def __init__(self, temperature: float) -> None:
        super().__init__()
        if not isfinite(temperature) or temperature <= 0:
            raise ValueError("temperature must be finite and positive")
        self.temperature = temperature

    def forward(self, features: Tensor, labels: Tensor, proxies: Tensor) -> Tensor:
        logits = F.normalize(features, dim=1) @ F.normalize(proxies[labels], dim=1).T
        log_probabilities = F.log_softmax(logits / self.temperature, dim=1)
        positives = labels[:, None] == labels[None, :]
        return -(log_probabilities * positives).sum(1).div(positives.sum(1)).mean()


class PCR(BatchClassifier):
    """Proxy-based Contrastive Replay (PCR).

    PCR [#pcr]_ trains an encoder and one learned proxy per schema class. Each
    update uses current and reservoir examples, with an original and augmented
    view of each. The contrastive targets are the proxies for labels in that
    combined batch, including duplicates. Prediction uses cosine similarity to
    all class proxies; softmax scores are not calibrated probabilities.

    The first batch fills memory without an optimizer update. Later batches
    use only historical examples for replay, then enter memory after all
    substeps. Replay uses the existing reservoir sampler with replacement.
    No task boundaries are required. Supply an augmentation suited to the data.
    ``seed`` controls the reservoir and proxy initialization; the caller seeds
    the encoder and any random augmentation.

    >>> from capymoa.ocl.strategy import PCR
    >>> schema = Schema.from_custom(
    ...     ["x0", "x1", "class"], "class", {"class": ["a", "b"]}
    ... )
    >>> encoder = nn.Linear(2, 8)
    >>> learner = PCR(schema, encoder, feature_dim=8, augment=lambda x: x.flip(-1))
    >>> learner.batch_train(torch.eye(2), torch.tensor([0, 1]))
    >>> learner.batch_train(torch.eye(2), torch.tensor([1, 0]))
    >>> learner.batch_predict_proba(torch.ones(3, 2)).shape
    torch.Size([3, 2])

    .. [#pcr] Lin, Huiwei, Baoquan Zhang, Shanshan Feng, Xutao Li, and Yunming Ye.
        "PCR: Proxy-Based Contrastive Replay for Online Class-Incremental
        Continual Learning." CVPR, 2023, pp. 24246-24255.
        https://openaccess.thecvf.com/content/CVPR2023/html/Lin_PCR_Proxy-Based_Contrastive_Replay_for_Online_Class-Incremental_Continual_Learning_CVPR_2023_paper.html
    """

    def __init__(
        self,
        schema: Schema,
        model: nn.Module,
        feature_dim: int,
        augment: Callable[[Tensor], Tensor],
        optimiser: Optimizer
        | Callable[[Iterator[nn.Parameter]], Optimizer]
        | None = None,
        *,
        feature_extractor: Callable[[Tensor], Tensor] | None = None,
        temperature: float = 0.09,
        buffer_capacity: int = 200,
        replay_batch_size: int | None = 10,
        substeps: int = 1,
        seed: int = 0,
        device: torch.device | str = "cpu",
    ) -> None:
        """Construct a PCR learner.

        :param schema: Fixed classification schema, including input shape.
        :param model: Encoder, or model exposing ``forward_features``/``features``.
            Otherwise the forward output is used as the embedding.
        :param feature_dim: Width of encoder features.
        :param augment: Callable preserving ``(batch, *schema.shape)``. Produces
            a second view; the first view is the original example.
        :param optimiser: Optimizer or factory receiving encoder and proxy
            parameters. None selects SGD with lr=0.1. For an instance, missing
            trainable parameters are added using its default options.
        :param feature_extractor: Optional callable using the model's parameters;
            overrides its feature methods and forward output.
        :param temperature: Finite positive scale for loss and inference logits.
        :param buffer_capacity: Maximum number of raw examples and labels stored.
        :param replay_batch_size: Replay samples per substep, or None to match
            the current batch size. Sampling is with replacement.
        :param substeps: Optimizer steps per batch once memory is non-empty.
        :param seed: Seed for reservoir sampling and proxy initialization.
        :param device: Device for the model, proxies and replay memory.
        """
        super().__init__(schema, seed)
        if not schema.is_classification():
            raise ValueError("PCR requires a classification schema")
        for name, value in (
            ("feature_dim", feature_dim),
            ("buffer_capacity", buffer_capacity),
            ("substeps", substeps),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if replay_batch_size is not None and (
            isinstance(replay_batch_size, bool)
            or not isinstance(replay_batch_size, int)
            or replay_batch_size <= 0
        ):
            raise ValueError("replay_batch_size must be a positive integer or None")
        if not callable(augment):
            raise TypeError("augment must be callable")
        if feature_extractor is not None and not callable(feature_extractor):
            raise TypeError("feature_extractor must be callable")
        self._loss = _ProxyContrastiveLoss(temperature)
        self.device = torch.device(device)
        self._model = model.to(self.device)
        self.x_dtype = next(model.parameters(), torch.empty(0)).dtype
        self._shape = tuple(schema.shape)
        self._num_features = schema.get_num_attributes()
        self._num_classes = schema.get_num_classes()
        self._feature_dim = feature_dim
        self._feature_extractor = feature_extractor
        self._augment = (
            augment.to(self.device) if isinstance(augment, nn.Module) else augment
        )
        # Seed the new head without advancing the caller's random generators.
        with torch.random.fork_rng(devices=[]):
            torch.set_rng_state(torch.Generator().manual_seed(seed).get_state())
            proxies = nn.Linear(feature_dim, self._num_classes, bias=False)
        self._proxies = proxies.to(device=self.device, dtype=self.x_dtype)
        parameters = [
            p
            for p in (*model.parameters(), *self._proxies.parameters())
            if p.requires_grad
        ]
        if optimiser is None:
            optimiser = torch.optim.SGD(parameters, lr=0.1)
        elif isinstance(optimiser, Optimizer):
            owned = {id(p) for group in optimiser.param_groups for p in group["params"]}
            missing = [p for p in parameters if id(p) not in owned]
            if missing:
                optimiser.add_param_group({"params": missing})
        elif callable(optimiser):
            optimiser = optimiser(iter(parameters))
        else:
            raise TypeError("optimiser must be an optimizer, factory or None")
        if not isinstance(optimiser, Optimizer):
            raise TypeError("optimiser factory must return an optimizer")
        owned = {id(p) for group in optimiser.param_groups for p in group["params"]}
        allowed = {id(p) for p in (*model.parameters(), *self._proxies.parameters())}
        if not {id(p) for p in parameters} <= owned or not owned <= allowed:
            raise ValueError("Optimizer must cover the encoder and proxies only")
        self._optimiser = optimiser
        self._buffer = ReservoirSampler(
            buffer_capacity,
            {"x": ((self._num_features,), self.x_dtype), "y": ((), self.y_dtype)},
            torch.Generator().manual_seed(seed),
        ).to(self.device)
        self._replay_batch_size = replay_batch_size
        self._substeps = substeps
        self.last_loss: float | None = None

    def _prepare_x(self, x: Tensor) -> Tensor:
        if x.ndim < 2 or x.shape[0] == 0:
            raise ValueError("x must contain a non-empty batch")
        flat = x.reshape(x.shape[0], -1)
        if flat.shape[1] != self._num_features:
            raise ValueError("feature count does not match the schema")
        return flat.to(device=self.device, dtype=self.x_dtype).reshape(-1, *self._shape)

    def _encode(self, x: Tensor) -> Tensor:
        extractor = self._feature_extractor
        if extractor is None:
            extractor = getattr(self._model, "forward_features", None)
        if extractor is None:
            extractor = getattr(self._model, "features", None)
        features = extractor(x) if extractor is not None else self._model(x)
        features = features.reshape(x.shape[0], -1)
        if features.shape[1] != self._feature_dim:
            raise ValueError("encoder output width does not match feature_dim")
        return features

    def batch_train(self, x: Tensor, y: Tensor) -> None:
        """Learn from historical replay, then store the incoming examples."""
        x = self._prepare_x(x)
        y = y.to(device=self.device, dtype=self.y_dtype)
        if y.shape != (x.shape[0],) or ((y < 0) | (y >= self._num_classes)).any():
            raise ValueError("y must contain one valid class index per example")
        self.last_loss = 0.0
        if self._buffer.count:
            self._model.train()
            self._proxies.train()
            n = (
                x.shape[0]
                if self._replay_batch_size is None
                else self._replay_batch_size
            )
            losses = []
            for _ in range(self._substeps):
                replay = self._buffer.sample(n)
                rx = replay["x"].reshape(-1, *self._shape)
                ax, arx = self._augment(x.clone()), self._augment(rx.clone())
                if ax.shape != x.shape or arx.shape != rx.shape:
                    raise ValueError("augment must preserve the batch shape")
                features = self._encode(torch.cat((rx, arx, x, ax)))
                labels = torch.cat((replay["y"], replay["y"], y, y))
                loss = self._loss(features, labels, self._proxies.weight)
                self._optimiser.zero_grad()
                loss.backward()
                self._optimiser.step()
                losses.append(loss.detach())
            self.last_loss = torch.stack(losses).mean().item()
        self._buffer.update(x=x.reshape(x.shape[0], -1), y=y)

    @torch.no_grad()
    def batch_predict_proba(self, x: Tensor) -> Tensor:
        """Return softmax scores over cosine similarities to all class proxies."""
        modes = {module: module.training for module in self._model.modules()}
        self._model.eval()
        try:
            features = F.normalize(self._encode(self._prepare_x(x)), dim=1)
            proxies = F.normalize(self._proxies.weight, dim=1)
            return (features @ proxies.T / self._loss.temperature).softmax(dim=1)
        finally:
            for module, training in modes.items():
                module.training = training

    def predict_proba(self, instance: Instance) -> LabelProbabilities:
        """Return single-instance probabilities on CPU, including for GPU models."""
        x = torch.as_tensor(instance.x).reshape(1, -1)
        return (
            self.batch_predict_proba(x)
            .to(device="cpu", dtype=torch.float64)
            .numpy()
            .reshape(-1)
        )

    def __str__(self) -> str:
        return f"PCR(temperature={self._loss.temperature}, buffer_capacity={self._buffer.capacity})"
