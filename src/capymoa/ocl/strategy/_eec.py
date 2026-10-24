"""Task-wise image replay with Encoding Episodes as Concepts."""

from collections.abc import Callable
from contextlib import contextmanager
from dataclasses import dataclass
from math import exp, prod

import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as F

from capymoa.base import BatchClassifier
from capymoa.core import Instance
from capymoa.ocl.evaluation.events import TestBegin, TrainTaskBegin, TrainTaskEnd
from capymoa.ocl.events import Dispatcher, Handler
from capymoa.ocl.util._optim import reset_optimizer_state
from capymoa.stream import Schema


@contextmanager
def _evaluation(*models: nn.Module):
    """Use evaluation mode and restore each submodule's original mode."""
    modes = {module: module.training for model in models for module in model.modules()}
    try:
        for model in models:
            model.eval()
        yield
    finally:
        for module, mode in modes.items():
            module.training = mode


class _Autoencoder(nn.Module):
    """The three-stage convolutional architecture in Appendix B of the paper."""

    def __init__(self, channels: int):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(channels, 64, 3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 32, 3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 16, 3, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(16, 32, 3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 64, 3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(64, channels, 3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.decoder(self.encoder(x))


@dataclass
class _Concept:
    mean: Tensor
    label: int
    task: int
    count: int = 1
    variance: Tensor | None = None

    @property
    def units(self) -> int:
        # An episode costs one latent vector; a centroid/covariance pair costs two.
        return 1 if self.variance is None else 2


def _merge(first: _Concept, second: _Concept) -> _Concept:
    """Pool sample means and diagonal sample covariances without raw episodes."""
    if (first.task, first.label) != (second.task, second.label):
        raise ValueError("Only concepts from the same task and class can be merged.")
    n, m = first.count, second.count
    delta = second.mean - first.mean
    mean = first.mean + delta * (m / (n + m))
    scatter = delta.square() * (n * m / (n + m))
    if first.variance is not None:
        scatter = scatter + (n - 1) * first.variance
    if second.variance is not None:
        scatter = scatter + (m - 1) * second.variance
    return _Concept(mean, first.label, first.task, n + m, scatter / (n + m - 1))


def _quotas(groups: list[list[_Concept]], budget: int) -> list[int]:
    """Round proportional per-class budgets, preserving each class distribution."""
    sizes = [sum(c.units for c in group) for group in groups]
    minima = [1 if sum(c.count for c in group) == 1 else 2 for group in groups]
    if budget < sum(minima):
        raise ValueError(
            "capacity cannot hold the new task and one distribution per old task/class. "
            "Increase capacity; an uncompressed episode costs 1 and a concept costs 2."
        )
    ideal = [size * budget / sum(sizes) for size in sizes]
    quotas = [max(low, min(size, int(q))) for low, size, q in zip(minima, sizes, ideal)]
    while sum(quotas) > budget:
        i = max(
            (i for i in range(len(groups)) if quotas[i] > minima[i]),
            key=lambda i: quotas[i] - ideal[i],
        )
        quotas[i] -= 1
    while sum(quotas) < min(budget, sum(sizes)):
        i = max(
            (i for i in range(len(groups)) if quotas[i] < sizes[i]),
            key=lambda i: ideal[i] - quotas[i],
        )
        quotas[i] += 1
    return quotas


class EEC(BatchClassifier, Handler):
    """Encoding Episodes as Concepts for task-wise class-incremental learning.

    EEC stores encoded images and replays their reconstructions. When latent memory
    fills, it merges nearby episodes of the same old task and class into Gaussian
    concepts. Pseudo-images are accepted only if the classifier predicts their
    source class. The autoencoder uses both pixel and classifier-feature losses.

    This implements the separate-autoencoder EEC variant from `Ayub and Wagner,
    ICLR 2021 <https://arxiv.org/abs/2101.04904>`_. It follows Algorithms 1--4 and
    the architecture in Appendix B. The released experiment script differs from
    the paper in its third convolution's stride and selects k-means; this class
    uses the paper's three downsampling stages and closest-pair integration.

    This is an offline task-wise method. ``batch_train`` caches the active task;
    ``TestBegin`` fits the classifier before evaluation, and ``TrainTaskEnd``
    trains its autoencoder, stores latent episodes and releases the raw cache.
    Use :func:`capymoa.ocl.evaluate_ocl` with ``epochs=1`` and
    ``continual_evaluations=1``. Further training after evaluation within a task
    raises an error. Predictions on training batches precede the offline fit.
    Direct use requires the same events through a
    :class:`~capymoa.ocl.events.Dispatcher`.

    ``capacity`` counts latent vectors, not bytes: an episode costs one vector,
    and a centroid plus diagonal covariance costs two. New task episodes remain
    uncompressed; older classes receive proportional budgets (Equation 4), with
    integer rounding and a minimum of one distribution per task/class. A budget
    that cannot hold these representations raises an error. Autoencoders and the
    active task cache are outside this budget and are reported separately by
    :meth:`memory_breakdown_bytes`. Autoencoder storage grows with task count.

    Equation 5 uses a reconstruction depth of one for each frozen per-task
    autoencoder, rather than elapsed task count. If original task accuracy is
    zero, its relative degradation is undefined; replay receives weight one.
    A finite rejection-sampling limit raises an error if too few valid
    pseudo-images can be generated, instead of accepting incorrect labels.

    The classifier may accept flat inputs (the default) or images. Its feature
    extractor must return the convolutional features used for content loss.
    Supplied data transforms and classifier architecture remain the caller's
    choice. The default autoencoder expects image dimensions divisible by eight.

    Example with small epoch counts for demonstration:

    >>> from capymoa.ocl import evaluate_ocl
    >>> from capymoa.ocl.datasets import TinySplitMNIST
    >>> scenario = TinySplitMNIST()
    >>> torch.manual_seed(0)
    <torch._C.Generator object at ...>
    >>> features = nn.Sequential(nn.Conv2d(1, 4, 3, padding=1), nn.ReLU())
    >>> model = nn.Sequential(features, nn.Flatten(), nn.Linear(4 * 16 * 16, 10))
    >>> learner = EEC(scenario.schema, model, feature_extractor=features,
    ...               reshape_input=True, epochs=1, subsequent_epochs=1,
    ...               autoencoder_epochs=1)
    >>> results = evaluate_ocl(learner, scenario.train_loaders(32),
    ...                        scenario.test_loaders(32))
    >>> results['accuracy_matrix'].shape
    (5, 5)
    """

    def __init__(
        self,
        schema: Schema,
        model: nn.Module,
        optimiser: torch.optim.Optimizer | None = None,
        *,
        capacity: int = 25000,
        feature_extractor: Callable[[Tensor], Tensor] | None = None,
        autoencoder_factory: Callable[[], nn.Module] | None = None,
        epochs: int = 200,
        subsequent_epochs: int = 45,
        autoencoder_epochs: int = 100,
        batch_size: int = 128,
        content_weight: float = 0.7,
        autoencoder_lr: float = 0.001,
        autoencoder_weight_decay: float = 0.0005,
        filter_attempts: int = 100,
        reshape_input: bool = False,
        device: str | torch.device = "cpu",
        seed: int = 0,
    ) -> None:
        """Construct an EEC learner.

        :param schema: Classification schema whose ``shape`` is ``(C, H, W)``.
        :param model: Classifier returning logits for all schema classes.
        :param optimiser: Classifier optimiser. Defaults to SGD with learning rate
            0.1, momentum 0.9 and weight decay 0.0005. State and starting learning
            rates are reset each task. The first task uses milestones 60, 120,
            160 and factor 0.2; later tasks use milestone 37 and factor 0.1.
        :param capacity: Maximum latent-vector units, excluding model parameters.
        :param feature_extractor: Classifier feature function. If omitted, uses
            ``model.forward_features`` or ``model.features``. It sees the same
            shaped input as ``model``. Required when ``content_weight > 0``.
        :param autoencoder_factory: Optional factory for a fresh trainable module
            with callable ``encoder`` and ``decoder`` attributes. Both accept
            batched tensors; the decoder must reconstruct ``schema.shape``.
        :param epochs: Classifier epochs for the first nonempty task.
        :param subsequent_epochs: Classifier epochs for later tasks.
        :param autoencoder_epochs: Epochs for each new autoencoder.
        :param batch_size: Mini-batch size for classifier and autoencoder fitting.
        :param content_weight: Weight of feature loss in Equation 3.
        :param autoencoder_lr: Adam learning rate, reduced tenfold after epoch 50.
        :param autoencoder_weight_decay: Adam weight decay.
        :param filter_attempts: Maximum rejection-sampling rounds per output batch.
            Failure to obtain enough correctly classified images raises an error;
            incorrectly classified images are never substituted.
        :param reshape_input: Pass images to the classifier instead of flat vectors.
        :param device: Compute device.
        :param seed: Seed for shuffling, latent sampling and autoencoder creation.
        """
        super().__init__(schema, random_seed=seed)
        for name, value in {
            "capacity": capacity,
            "epochs": epochs,
            "subsequent_epochs": subsequent_epochs,
            "autoencoder_epochs": autoencoder_epochs,
            "batch_size": batch_size,
            "filter_attempts": filter_attempts,
        }.items():
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer.")
        if not schema.is_classification():
            raise ValueError("EEC requires a classification schema.")
        self._shape = tuple(schema.shape)
        if len(self._shape) != 3:
            raise ValueError("schema.shape must be (channels, height, width).")
        if autoencoder_factory is None and any(d % 8 for d in self._shape[1:]):
            raise ValueError(
                "Default autoencoder needs height and width divisible by 8."
            )
        if not 0 <= content_weight <= 1:
            raise ValueError("content_weight must be between zero and one.")
        if not np.isfinite(autoencoder_lr) or autoencoder_lr <= 0:
            raise ValueError("autoencoder_lr must be finite and positive.")
        if not np.isfinite(autoencoder_weight_decay) or autoencoder_weight_decay < 0:
            raise ValueError("autoencoder_weight_decay must be finite and nonnegative.")
        if feature_extractor is None:
            feature_extractor = getattr(model, "forward_features", None)
            if feature_extractor is None:
                feature_extractor = getattr(model, "features", None)
        if content_weight and not callable(feature_extractor):
            raise ValueError("Provide a classifier feature_extractor for content loss.")
        self.device = torch.device(device)
        self.model = model.to(self.device)
        self.optimiser = optimiser or torch.optim.SGD(
            self.model.parameters(), lr=0.1, momentum=0.9, weight_decay=0.0005
        )
        self._starting_lrs = [group["lr"] for group in self.optimiser.param_groups]
        self.capacity = capacity
        self.epochs = epochs
        self.subsequent_epochs = subsequent_epochs
        self.autoencoder_epochs = autoencoder_epochs
        self.batch_size = batch_size
        self.content_weight = content_weight
        self._feature_extractor = feature_extractor
        self._autoencoder_factory = autoencoder_factory
        self._autoencoder_lr = autoencoder_lr
        self._autoencoder_weight_decay = autoencoder_weight_decay
        self._filter_attempts = filter_attempts
        self._reshape_input = reshape_input
        self._generator = torch.Generator().manual_seed(seed)
        self._autoencoders: dict[int, nn.Module] = {}
        self._concepts: list[_Concept] = []
        self._original_accuracy: dict[int, float] = {}
        self._active_task: int | None = None
        self._fitted = False
        self._task_x: list[Tensor] = []
        self._task_y: list[Tensor] = []

    def attach_with(self, source: Dispatcher) -> "EEC":
        source.subscribe(TrainTaskBegin, self._begin_task)
        source.subscribe(TestBegin, self._before_test)
        source.subscribe(TrainTaskEnd, self._end_task)
        return self

    def _begin_task(self, event: TrainTaskBegin) -> None:
        if self._active_task is not None:
            raise RuntimeError("Finish the active task with TrainTaskEnd first.")
        if event.train_task in self._autoencoders:
            raise ValueError("EEC needs a new task ID for each increment.")
        self._active_task = event.train_task
        self._fitted = False

    def batch_train(self, x: Tensor, y: Tensor) -> None:
        """Cache active-task images for offline fitting before task evaluation."""
        if self._active_task is None:
            raise RuntimeError("EEC requires TrainTaskBegin and TrainTaskEnd events.")
        if self._fitted:
            raise RuntimeError(
                "EEC cannot receive more data after task evaluation. "
                "Use evaluate_ocl with epochs=1 and continual_evaluations=1."
            )
        if x.numel() != x.shape[0] * prod(self._shape) or y.shape != (x.shape[0],):
            raise ValueError(
                "Batch shapes must match schema.shape and one label per image."
            )
        if not x.shape[0]:
            return
        if not torch.isfinite(x).all():
            raise ValueError("EEC inputs must be finite.")
        if (
            y.dtype != torch.long
            or y.min() < 0
            or y.max() >= self.schema.get_num_classes()
        ):
            raise ValueError("Labels must be int64 indices in the schema class range.")
        self._task_x.append(
            x.detach()
            .to(device="cpu", dtype=self.x_dtype)
            .reshape(-1, *self._shape)
            .clone()
        )
        self._task_y.append(y.detach().cpu().clone())

    def _model_input(self, x: Tensor) -> Tensor:
        x = x.to(self.device, dtype=self.x_dtype).reshape(-1, *self._shape)
        return x if self._reshape_input else x.flatten(1)

    @torch.no_grad()
    def batch_predict_proba(self, x: Tensor) -> Tensor:
        with _evaluation(self.model):
            return self.model(self._model_input(x)).softmax(dim=1)

    def predict_proba(self, instance: Instance) -> np.ndarray:
        x = torch.from_numpy(instance.x.copy()).reshape(1, -1)
        return self.batch_predict_proba(x)[0].cpu().numpy().astype(np.float64)

    @torch.no_grad()
    def _accuracy(self, x: Tensor, y: Tensor) -> float:
        correct = 0
        for start in range(0, len(x), self.batch_size):
            predictions = self.batch_predict(x[start : start + self.batch_size]).cpu()
            correct += int(
                (predictions == y[start : start + self.batch_size].cpu()).sum()
            )
        return correct / len(x)

    @torch.no_grad()
    def _decode(self, task: int, latents: Tensor) -> Tensor:
        autoencoder = self._autoencoders[task]
        with _evaluation(autoencoder):
            images = autoencoder.decoder(latents.to(self.device))
        if (
            images.shape != (len(latents), *self._shape)
            or not torch.isfinite(images).all()
        ):
            raise ValueError(
                "Autoencoder decoder must return finite images matching schema.shape."
            )
        return images.detach().cpu()

    @torch.no_grad()
    def _pseudo_images(self, concept: _Concept) -> Tensor:
        assert concept.variance is not None
        accepted: list[Tensor] = []
        remaining = concept.count
        while remaining:
            needed = min(self.batch_size, remaining)
            for _ in range(self._filter_attempts):
                count = min(self.batch_size, 2 * needed)
                noise = torch.randn(
                    (count, *concept.mean.shape), generator=self._generator
                )
                latent = concept.mean + noise * concept.variance.clamp_min(0).sqrt()
                images = self._decode(concept.task, latent)
                matches = self.batch_predict(images).cpu() == concept.label
                images = images[matches][:needed]
                accepted.append(images)
                needed -= len(images)
                remaining -= len(images)
                if needed == 0:
                    break
            if needed:
                break
        if remaining == 0:
            return torch.cat(accepted)
        raise RuntimeError(
            f"EEC could not generate {concept.count} accepted pseudo-images for "
            f"task {concept.task}, class {concept.label}. Increase filter_attempts "
            "or improve classifier/autoencoder training. No mismatched image was used."
        )

    def _replay(self) -> list[tuple[Tensor, Tensor, float]]:
        """Build fixed replay sets and Equation 5 weights before changing the classifier."""
        sets: list[tuple[Tensor, Tensor, float]] = []
        for task, original_accuracy in self._original_accuracy.items():
            for pseudo in (False, True):
                concepts = [
                    c
                    for c in self._concepts
                    if c.task == task and (c.variance is not None) == pseudo
                ]
                if not concepts:
                    continue
                if pseudo:
                    x = torch.cat([self._pseudo_images(c) for c in concepts])
                    y = torch.tensor(
                        [c.label for c in concepts for _ in range(c.count)]
                    )
                else:
                    x = torch.cat(
                        [
                            self._decode(
                                task,
                                torch.stack(
                                    [
                                        c.mean
                                        for c in concepts[
                                            start : start + self.batch_size
                                        ]
                                    ]
                                ),
                            )
                            for start in range(0, len(concepts), self.batch_size)
                        ]
                    )
                    y = torch.tensor([c.label for c in concepts])
                degradation = (
                    max(0.0, min(1.0, 1 - self._accuracy(x, y) / original_accuracy))
                    if original_accuracy > 0
                    else 0.0
                )
                # Separate frozen autoencoders reconstruct each task once; this
                # exponent is reconstruction depth, never elapsed task count.
                sets.append((x, y, exp(-degradation)))
        return sets

    def _before_test(self, event: TestBegin) -> None:
        self._fit_classifier()

    def _fit_classifier(self) -> None:
        if self._fitted or not self._task_x:
            return
        task_x, task_y = torch.cat(self._task_x), torch.cat(self._task_y)
        replay = self._replay()
        sets = [(task_x, task_y, 1.0), *replay]
        reset_optimizer_state(self.optimiser)
        for group, lr in zip(
            self.optimiser.param_groups, self._starting_lrs, strict=True
        ):
            group["lr"] = lr
        first = not self._autoencoders
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            self.optimiser,
            milestones=[60, 120, 160] if first else [37],
            gamma=0.2 if first else 0.1,
        )
        # Stratify mini-batches by task and source, so each contributes its own
        # mean CE (Equation 6) rather than being diluted by a pooled replay mean.
        for _ in range(self.epochs if first else self.subsequent_epochs):
            self.model.train()
            orders = [
                torch.randperm(len(x), generator=self._generator) for x, _, _ in sets
            ]
            steps = max(
                (len(x) + self.batch_size - 1) // self.batch_size for x, _, _ in sets
            )
            for step in range(steps):
                self.optimiser.zero_grad()
                for i, (x, y, weight) in enumerate(sets):
                    batches = (len(x) + self.batch_size - 1) // self.batch_size
                    if step and step % batches == 0:
                        orders[i] = torch.randperm(len(x), generator=self._generator)
                    start = (step % batches) * self.batch_size
                    indices = orders[i][start : start + self.batch_size]
                    logits = self.model(self._model_input(x[indices]))
                    loss = F.cross_entropy(logits, y[indices].to(self.device))
                    (weight * loss).backward()
                self.optimiser.step()
            scheduler.step()
        assert self._active_task is not None
        self._original_accuracy[self._active_task] = self._accuracy(task_x, task_y)
        self._fitted = True

    def _new_autoencoder(self) -> nn.Module:
        # Factory initialisation must not disturb the application's random state.
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(
                int(torch.randint(2**31, (), generator=self._generator))
            )
            autoencoder = (
                self._autoencoder_factory()
                if self._autoencoder_factory is not None
                else _Autoencoder(self._shape[0])
            )
        if not isinstance(autoencoder, nn.Module) or not all(
            callable(getattr(autoencoder, name, None))
            for name in ("encoder", "decoder")
        ):
            raise ValueError(
                "autoencoder_factory must return a module with encoder and decoder."
            )
        if any(autoencoder is previous for previous in self._autoencoders.values()):
            raise ValueError(
                "autoencoder_factory must return a fresh module for each task."
            )
        return autoencoder.to(self.device)

    def _train_autoencoder(self, task_x: Tensor) -> nn.Module:
        autoencoder = self._new_autoencoder()
        optimizer = torch.optim.Adam(
            autoencoder.parameters(),
            lr=self._autoencoder_lr,
            weight_decay=self._autoencoder_weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [50], gamma=0.1)
        extractor = self._feature_extractor
        fixed = [self.model]
        if isinstance(extractor, nn.Module):
            fixed.append(extractor)
        flags = {p: p.requires_grad for model in fixed for p in model.parameters()}
        try:
            with _evaluation(*fixed):
                for p in flags:
                    p.requires_grad_(False)
                for _ in range(self.autoencoder_epochs):
                    autoencoder.train()
                    order = torch.randperm(len(task_x), generator=self._generator)
                    for start in range(0, len(task_x), self.batch_size):
                        x = task_x[order[start : start + self.batch_size]].to(
                            self.device
                        )
                        # BatchNorm cannot estimate variance from one value/channel.
                        if len(x) == 1:
                            for module in autoencoder.modules():
                                if isinstance(module, nn.modules.batchnorm._BatchNorm):
                                    module.eval()
                        optimizer.zero_grad()
                        reconstructed = autoencoder.decoder(autoencoder.encoder(x))
                        if reconstructed.shape != x.shape:
                            raise ValueError(
                                "Autoencoder must reconstruct schema.shape exactly."
                            )
                        loss = (1 - self.content_weight) * F.mse_loss(reconstructed, x)
                        if self.content_weight:
                            assert extractor is not None
                            with torch.no_grad():
                                target = extractor(self._model_input(x))
                            features = extractor(self._model_input(reconstructed))
                            loss = loss + self.content_weight * F.mse_loss(
                                features, target
                            )
                        loss.backward()
                        optimizer.step()
                    scheduler.step()
        finally:
            for p, flag in flags.items():
                p.requires_grad_(flag)
        autoencoder.eval().requires_grad_(False)
        return autoencoder

    def _integrate(self, episodes: list[_Concept]) -> list[_Concept]:
        if self.memory_units + len(episodes) <= self.capacity:
            return self._concepts + episodes
        groups: dict[tuple[int, int], list[_Concept]] = {}
        for concept in self._concepts:
            groups.setdefault((concept.task, concept.label), []).append(concept)
        if not groups:
            raise ValueError(
                "capacity must hold all encoded episodes of the first task."
            )
        old = list(groups.values())
        quotas = _quotas(old, self.capacity - len(episodes))
        reduced: list[_Concept] = []
        for group, quota in zip(old, quotas, strict=True):
            group = group.copy()
            while sum(c.units for c in group) > quota:
                means = torch.stack([c.mean.flatten() for c in group])
                distances = torch.cdist(means, means)
                distances.fill_diagonal_(float("inf"))
                pair = int(distances.argmin())
                a, b = sorted(divmod(pair, len(group)))
                group[a] = _merge(group[a], group[b])
                del group[b]
            reduced.extend(group)
        return reduced + episodes

    def _end_task(self, event: TrainTaskEnd) -> None:
        if self._active_task is None:
            return
        if event.train_task != self._active_task:
            raise ValueError("TrainTaskEnd must match the active task.")
        if self._task_x:
            self._fit_classifier()
            x, y = torch.cat(self._task_x), torch.cat(self._task_y)
            autoencoder = self._train_autoencoder(x)
            episodes: list[_Concept] = []
            with torch.no_grad():
                for start in range(0, len(x), self.batch_size):
                    latent = autoencoder.encoder(
                        x[start : start + self.batch_size].to(self.device)
                    ).cpu()
                    if not torch.isfinite(latent).all():
                        raise ValueError(
                            "Autoencoder must produce finite latent episodes."
                        )
                    episodes.extend(
                        _Concept(code.clone(), int(label), self._active_task)
                        for code, label in zip(
                            latent, y[start : start + self.batch_size], strict=True
                        )
                    )
            # Integrate transactionally: a failed budget check keeps old memory.
            concepts = self._integrate(episodes)
            self._concepts = concepts
            self._autoencoders[self._active_task] = autoencoder
        self._task_x.clear()
        self._task_y.clear()
        self._active_task = None

    @property
    def memory_units(self) -> int:
        """Latent-vector units used by episodes and Gaussian concepts."""
        return sum(c.units for c in self._concepts)

    def memory_breakdown_bytes(self) -> dict[str, int]:
        """Report tensor storage, excluding Python objects, optimiser and workspaces."""

        def size(x: Tensor) -> int:
            return x.numel() * x.element_size()

        latent = sum(
            size(c.mean) + (0 if c.variance is None else size(c.variance))
            for c in self._concepts
        )
        autoencoders = sum(
            size(x)
            for ae in self._autoencoders.values()
            for x in (*ae.parameters(), *ae.buffers())
        )
        classifier = sum(
            size(x) for x in (*self.model.parameters(), *self.model.buffers())
        )
        cache = sum(size(x) for x in (*self._task_x, *self._task_y))
        return {
            "latent": latent,
            "autoencoders": autoencoders,
            "classifier": classifier,
            "active_task": cache,
            "total": latent + autoencoders + classifier + cache,
        }
