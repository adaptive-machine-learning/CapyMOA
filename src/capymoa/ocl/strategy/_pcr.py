"""Proxy-based Contrastive Replay."""

from collections.abc import Callable
from itertools import chain
from math import isfinite, sqrt
from typing import Any

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch.optim import Optimizer

from capymoa.base import BatchClassifier, TorchMixin
from capymoa.core.torch import FeatureModel
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


class _PCRModule(nn.Module):
    """Torch state of PCR."""

    seen_classes: Tensor

    def __init__(
        self,
        model: FeatureModel,
        proxies: nn.Linear,
        buffer: ReservoirSampler,
        augment: Callable[[Tensor], Tensor],
    ) -> None:
        super().__init__()
        self.model = model
        self.proxies = proxies
        self.buffer = buffer
        self.augment = augment
        self.register_buffer(
            "seen_classes", torch.zeros(proxies.out_features, dtype=torch.bool)
        )


class PCR(TorchMixin[_PCRModule], BatchClassifier):
    """Proxy-based Contrastive Replay (PCR).

    PCR [#pcr]_ trains an encoder and one learned proxy per class. Each update uses current
    and reservoir examples, with an original and augmented view of each. The contrastive
    targets are the proxies for labels in that combined batch. Prediction uses cosine
    similarity to all class proxies.

    Other implementations [#pcr-original]_ [#pcr-mkd]_.

    >>> import torch
    >>> from torch import nn
    >>> from capymoa.core.torch import FeatureModel
    >>> from capymoa.ocl.strategy import PCR
    >>> from capymoa.stream import Schema
    >>> schema = Schema.from_custom(
    ...     ["x0", "x1", "class"], "class", {"class": ["a", "b"]}
    ... )
    >>> class Encoder(FeatureModel):
    ...     def __init__(self):
    ...         super().__init__()
    ...         self.encoder = nn.Linear(2, 8)
    ...     def features(self, x):
    ...         return self.encoder(x)
    ...     def forward(self, x):
    ...         return self.features(x)
    >>> learner = PCR(schema, Encoder(), feature_dim=8, augment=lambda x: x.flip(-1))
    >>> learner.batch_train(torch.eye(2), torch.tensor([0, 1]))
    >>> learner.batch_train(torch.eye(2), torch.tensor([1, 0]))
    >>> learner.batch_predict_proba(torch.ones(3, 2)).shape
    torch.Size([3, 2])

    .. [#pcr] Lin, Huiwei, Baoquan Zhang, Shanshan Feng, Xutao Li, and Yunming Ye. "PCR:
        Proxy-Based Contrastive Replay for Online Class-Incremental Continual Learning."
        CVPR, 2023, pp. 24246-24255.
        https://openaccess.thecvf.com/content/CVPR2023/html/Lin_PCR_Proxy-Based_Contrastive_Replay_for_Online_Class-Incremental_Continual_Learning_CVPR_2023_paper.html

    .. [#pcr-original] FelixHuiweiLin. "PCR: Proxy-based Contrastive Replay for Online
       Class-Incremental Continual Learning." https://github.com/FelixHuiweiLin/PCR

    .. [#pcr-mkd] Nicolas1203. "PCR learner in mkd_ocl."
       https://github.com/Nicolas1203/mkd_ocl/blob/main/src/learners/baselines/pcr.py
    """

    def __init__(
        self,
        schema: Schema,
        model: FeatureModel,
        feature_dim: int,
        augment: Callable[[Tensor], Tensor],
        *,
        optimiser_class: type[Optimizer] = torch.optim.SGD,
        optimiser_kwargs: dict[str, Any] | None = None,
        temperature: float = 0.09,
        buffer_capacity: int = 200,
        replay_batch_size: int | None = 10,
        substeps: int = 1,
        seed: None | int = None,
        device: torch.device | str = "cpu",
    ) -> None:
        """Construct a PCR learner.

        :param schema: Fixed classification schema, including input shape.
        :param model: A :class:`~capymoa.core.torch.FeatureModel`. PCR uses its
            ``features`` method, not its classifier output.
        :param feature_dim: Width of flattened model features.
        :param augment: Callable preserving ``(batch, *schema.shape)``. Produces
            a second view; the first view is the original example.
        :param optimiser_class: Optimizer class for encoder and proxy parameters.
            Defaults to SGD.
        :param optimiser_kwargs: Keyword arguments passed to the optimizer.
            Defaults to ``{"lr": 0.001}`` when omitted or empty.
        :param temperature: Finite positive scale for loss and inference logits.
        :param buffer_capacity: Maximum number of raw examples and labels stored.
        :param replay_batch_size: Replay samples per substep, or None to match
            the current batch size. Sampling is with replacement.
        :param substeps: Optimizer steps per batch once memory is non-empty.
        :param seed: Seed for reservoir sampling and proxy initialization. If
            None, a seed is drawn from NumPy's global random generator.
        :param device: Device for the model, proxies and replay memory.
        """
        super().__init__(schema, seed)

        if not schema.is_classification():
            raise ValueError("PCR requires a classification schema")
        if not isinstance(model, FeatureModel):
            raise TypeError("model must be a FeatureModel")
        if feature_dim <= 0:
            raise ValueError("feature_dim must be positive")
        if buffer_capacity <= 0:
            raise ValueError("buffer_capacity must be positive")
        if substeps <= 0:
            raise ValueError("substeps must be positive")
        if replay_batch_size is not None and replay_batch_size <= 0:
            raise ValueError("replay_batch_size must be positive or None")
        if not callable(augment):
            raise TypeError("augment must be callable")

        self._loss = _ProxyContrastiveLoss(temperature)
        self._shape = tuple(schema.shape)
        self._num_features = schema.get_num_attributes()
        self._num_classes = schema.get_num_classes()
        self._feature_dim = feature_dim
        self.x_dtype = next(model.parameters(), torch.empty(0)).dtype

        # Initialize the class proxies with a reproducible random seed.
        proxies = nn.Linear(feature_dim, self._num_classes, bias=False)
        proxy_rng = torch.Generator().manual_seed(self.random_seed)
        nn.init.kaiming_uniform_(proxies.weight, a=sqrt(5), generator=proxy_rng)

        # Allocate the replay buffer and module.
        buffer = ReservoirSampler(
            buffer_capacity,
            {"x": ((self._num_features,), self.x_dtype), "y": ((), self.y_dtype)},
            torch.Generator().manual_seed(self.random_seed),
        )
        self._module = _PCRModule(model, proxies.to(self.x_dtype), buffer, augment)
        self._module = self._module.to(device)

        # Train the encoder and class proxies together.
        parameters = (
            param
            for param in chain(
                self._module.model.parameters(),
                self._module.proxies.parameters(),
            )
            if param.requires_grad
        )
        optimiser_kwargs = dict(optimiser_kwargs or {"lr": 1e-3})
        self._optimiser = optimiser_class(parameters, **optimiser_kwargs)

        self._replay_batch_size = replay_batch_size
        self._substeps = substeps
        self._last_loss: float | None = None

    def _prepare_x(self, x: Tensor) -> Tensor:
        if x.ndim < 2 or x.shape[0] == 0:
            raise ValueError("x must contain a non-empty batch")
        flat = x.reshape(x.shape[0], -1)
        if flat.shape[1] != self._num_features:
            raise ValueError("feature count does not match the schema")
        return flat.reshape(-1, *self._shape)

    def _encode(self, x: Tensor) -> Tensor:
        features = self._module.model.features(x)
        features = features.reshape(x.shape[0], -1)
        if features.shape[1] != self._feature_dim:
            raise ValueError("encoder output width does not match feature_dim")
        return features

    def batch_train(self, x: Tensor, y: Tensor) -> None:
        """Learn from historical replay, then store the incoming examples."""

        # Validate the incoming batch.
        x = self._prepare_x(x)
        if y.shape != (x.shape[0],) or ((y < 0) | (y >= self._num_classes)).any():
            raise ValueError("y must contain one valid class index per example")
        self._last_loss = 0.0

        # Fill memory before replay is available.
        if not self._module.buffer.count:
            self._module.buffer.update(x=x.reshape(x.shape[0], -1), y=y)
            self._module.seen_classes[y] = True
            return

        # Prepare the current batch and its augmented view.
        self._module.model.train()
        n = x.shape[0] if self._replay_batch_size is None else self._replay_batch_size
        ax = self._module.augment(x.clone())
        if ax.shape != x.shape:
            raise ValueError("augment must preserve the batch shape")

        # Replay history and update the encoder and proxies.
        losses = []
        for _ in range(self._substeps):
            replay = self._module.buffer.sample(n)
            rx = replay["x"].reshape(-1, *self._shape)
            arx = self._module.augment(rx.clone())
            if arx.shape != rx.shape:
                raise ValueError("augment must preserve the batch shape")
            features = self._encode(torch.cat((rx, arx, x, ax)))
            labels = torch.cat((replay["y"], replay["y"], y, y))
            loss = self._loss(features, labels, self._module.proxies.weight)
            self._optimiser.zero_grad()
            loss.backward()
            self._optimiser.step()
            losses.append(loss.detach())
        self._last_loss = torch.stack(losses).mean().item()

        # Save this batch for future replay and prediction masking.
        self._module.buffer.update(x=x.reshape(x.shape[0], -1), y=y)
        self._module.seen_classes[y] = True

    @torch.no_grad()
    def batch_predict_proba(self, x: Tensor) -> Tensor:
        """Return softmax scores for observed class proxies in evaluation mode."""
        x = self._prepare_x(x)
        self._module.model.eval()
        if not self._module.seen_classes.any():
            return x.new_full((x.shape[0], self._num_classes), 1 / self._num_classes)
        features = F.normalize(self._encode(x), dim=1)
        proxies = F.normalize(self._module.proxies.weight, dim=1)
        logits = features @ proxies.T / self._loss.temperature
        logits = logits.masked_fill(~self._module.seen_classes, -torch.inf)
        return logits.softmax(dim=1).to(x.dtype)

    def __str__(self) -> str:
        return f"PCR(temperature={self._loss.temperature}, buffer_capacity={self._module.buffer.capacity})"
