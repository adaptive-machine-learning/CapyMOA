"""Supervised Contrastive Replay using the existing CapyMOA OCL cores."""

from collections.abc import Callable, Iterator
from math import isfinite

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.optim import Optimizer

from capymoa.base import BatchClassifier
from capymoa.core import Instance, LabelProbabilities
from capymoa.ocl.evaluation.events import TrainTaskEnd
from capymoa.ocl.events import Dispatcher, Handler
from capymoa.ocl.util._replay import ReservoirSampler
from capymoa.stream import Schema


class _SupervisedContrastiveLoss(nn.Module):
    """Contrast each anchor with all other views, using same-label positives."""

    def __init__(self, temperature: float) -> None:
        super().__init__()
        if not isfinite(temperature) or temperature <= 0:
            raise ValueError("temperature must be finite and positive")
        self.temperature = temperature

    def forward(self, features: Tensor, labels: Tensor) -> Tensor:
        if features.ndim != 3 or features.shape[1] < 2:
            raise ValueError("features must have shape (batch, views >= 2, features)")
        if features.shape[0] == 0 or labels.shape != (features.shape[0],):
            raise ValueError("labels must match the non-empty feature batch")
        views = features.shape[1]
        embeddings = torch.cat(features.unbind(dim=1), dim=0)
        logits = embeddings @ embeddings.T / self.temperature
        logits = logits - logits.max(dim=1, keepdim=True).values.detach()
        diagonal = torch.eye(logits.shape[0], device=logits.device, dtype=torch.bool)
        positives = (labels[:, None] == labels[None, :]).repeat(views, views)
        positives = positives & ~diagonal
        log_prob = logits - torch.logsumexp(
            logits.masked_fill(diagonal, -torch.inf), dim=1, keepdim=True
        )
        return -(log_prob * positives).sum(1).div(positives.sum(1)).mean()


class SCR(BatchClassifier, Handler):
    """Supervised Contrastive Replay (SCR) [#scr]_.

    Trains an encoder and projection head on current and replay examples, using
    each example and an augmented view. At inference, normalized encoder features
    are compared with normalized class means recomputed from reservoir memory.
    The projection head is used for training only.

    Uses the existing schema, batch classifier, named reservoir and task events.
    Replay is sampled with replacement, using the reservoir's seeded generator.
    Incoming examples enter memory after all updates on that batch. There is no
    task-boundary requirement by default. Optional memory review uses
    ``TrainTaskEnd`` events, including for the final task; it happens after the
    loop's last evaluation of that task and does not change already reported metrics.

    Supply an augmentation appropriate to the data, as for RAR. No additional
    augmentation package is required. ``seed`` controls memory sampling and
    projection initialization; the caller seeds its model and random augmentation.
    Features and class labels must fit a fixed classification schema.

    >>> from capymoa.ocl.strategy import SCR
    >>> from capymoa.stream import Schema
    >>> import torch
    >>> from torch import nn
    >>> schema = Schema.from_custom(
    ...     ["x0", "x1", "class"], "class", {"class": ["a", "b"]}
    ... )
    >>> encoder = nn.Sequential(nn.Linear(2, 8), nn.ReLU())
    >>> learner = SCR(schema, encoder, feature_dim=8, augment=lambda x: x.flip(-1))
    >>> learner.batch_train(torch.eye(2), torch.tensor([0, 1]))
    >>> learner.batch_predict_proba(torch.ones(3, 2)).shape
    torch.Size([3, 2])

    .. [#scr] Mai, Zheda, Ruiwen Li, Hyunwoo Kim, and Scott Sanner.
        "Supervised Contrastive Replay: Revisiting the Nearest Class Mean
        Classifier in Online Class-Incremental Continual Learning."
        CVPR Workshops, 2021, pp. 3589-3599.
        https://openaccess.thecvf.com/content/CVPR2021W/CLVision/html/Mai_Supervised_Contrastive_Replay_Revisiting_the_Nearest_Class_Mean_Classifier_in_CVPRW_2021_paper.html
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
        projection_dim: int = 128,
        projection_head: str = "mlp",
        temperature: float = 0.1,
        buffer_capacity: int = 200,
        replay_batch_size: int | None = 100,
        substeps: int = 1,
        feature_batch_size: int = 128,
        ncm_temperature: float = 1.0,
        review_at_task_boundary: bool = False,
        review_batch_size: int = 100,
        review_gradient_scale: float = 0.1,
        seed: int = 0,
        device: torch.device | str = "cpu",
    ) -> None:
        """Construct an SCR learner.

        :param schema: Fixed classification schema, including input shape.
        :param model: Encoder, or a model exposing ``forward_features``/``features``.
            Otherwise its forward output is taken as the embedding. The caller
            controls model initialization, as for DER.
        :param feature_dim: Encoder embedding width, before the projection head.
        :param augment: Callable preserving ``(batch, *schema.shape)``. It makes
            the second view; the first is the unaugmented example.
        :param optimiser: Optimizer or factory; None selects SGD with lr=0.1.
            A factory receives model and
            projection parameters. With an instance, missing trainable parameters
            are added using its default options so the projection head is trained.
        :param feature_extractor: Optional embedding callable using ``model``'s
            parameters; overrides the model's feature methods and forward output.
        :param projection_dim: Width of contrastive embeddings.
        :param projection_head: ``mlp``, ``linear`` or ``identity``. Identity
            requires ``feature_dim == projection_dim``.
        :param temperature: Positive supervised contrastive loss temperature.
        :param buffer_capacity: Maximum number of raw examples and labels stored.
        :param replay_batch_size: Replay samples per update, or None for the
            current batch size. Sampling follows CapyMOA's replacement convention.
        :param substeps: Number of contrastive optimization steps per stream batch.
        :param feature_batch_size: Chunk size for encoding memory during inference.
        :param ncm_temperature: Positive scale for distance-based softmax scores;
            these scores are not calibrated class probabilities.
        :param review_at_task_boundary: Enable an extra memory pass on task-end
            events. False keeps the method task-free.
        :param review_batch_size: Batch size of optional task-end review.
        :param review_gradient_scale: Positive multiplier on the review loss.
        :param seed: Seed for replay sampling and projection-head initialization.
        :param device: Device for the model, projection and replay buffer.
        """
        super().__init__(schema, seed)
        if not schema.is_classification():
            raise ValueError("SCR requires a classification schema")
        for name, value in (
            ("feature_dim", feature_dim),
            ("projection_dim", projection_dim),
            ("buffer_capacity", buffer_capacity),
            ("substeps", substeps),
            ("feature_batch_size", feature_batch_size),
            ("review_batch_size", review_batch_size),
        ):
            if not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if replay_batch_size is not None and (
            not isinstance(replay_batch_size, int) or replay_batch_size <= 0
        ):
            raise ValueError("replay_batch_size must be positive or None")
        for name, value in (
            ("ncm_temperature", ncm_temperature),
            ("review_gradient_scale", review_gradient_scale),
        ):
            if not isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if not callable(augment):
            raise TypeError("augment must be callable")
        self._loss = _SupervisedContrastiveLoss(temperature)
        self.device = torch.device(device)
        self._model = model.to(self.device)
        self.x_dtype = next(model.parameters(), torch.empty(0)).dtype
        self._feature_dim = feature_dim
        self._feature_extractor = feature_extractor
        self._augment = (
            augment.to(self.device) if isinstance(augment, nn.Module) else augment
        )
        self._shape = tuple(schema.shape)
        self._num_features = schema.get_num_attributes()
        self._num_classes = schema.get_num_classes()
        # Initialize on CPU under a private RNG state without changing caller RNGs.
        with torch.random.fork_rng(devices=[]):
            torch.set_rng_state(torch.Generator().manual_seed(seed).get_state())
            if projection_head == "mlp":
                head = nn.Sequential(
                    nn.Linear(feature_dim, feature_dim),
                    nn.ReLU(),
                    nn.Linear(feature_dim, projection_dim),
                )
            elif projection_head == "linear":
                head = nn.Linear(feature_dim, projection_dim)
            elif projection_head == "identity" and feature_dim == projection_dim:
                head = nn.Identity()
            else:
                raise ValueError(
                    "projection_head must be mlp, linear, or identity with matching dimensions"
                )
        self._projection = head.to(device=self.device, dtype=self.x_dtype)
        parameters = [
            p for p in (*model.parameters(), *head.parameters()) if p.requires_grad
        ]
        if optimiser is None:
            self._optimiser = torch.optim.SGD(parameters, lr=0.1)
        elif isinstance(optimiser, Optimizer):
            owned = {id(p) for group in optimiser.param_groups for p in group["params"]}
            missing = [p for p in parameters if id(p) not in owned]
            if missing:
                optimiser.add_param_group({"params": missing})
            self._optimiser = optimiser
        else:
            self._optimiser = optimiser(iter(parameters))
        self._buffer = ReservoirSampler(
            buffer_capacity,
            {"x": ((self._num_features,), self.x_dtype), "y": ((), torch.long)},
            torch.Generator().manual_seed(seed),
        ).to(self.device)
        self._substeps = substeps
        self._replay_batch_size = replay_batch_size
        self._feature_batch_size = feature_batch_size
        self._ncm_temperature = ncm_temperature
        self._review = review_at_task_boundary
        self._review_batch_size = review_batch_size
        self._review_scale = review_gradient_scale
        self._prototype_cache: tuple[Tensor, Tensor] | None = None
        self._dispatcher: Dispatcher | None = None
        self.last_loss: float | None = None
        self.completed_reviews = 0

    def _prepare_x(self, x: Tensor) -> Tensor:
        if x.ndim < 2 or x.shape[0] == 0:
            raise ValueError("x must contain a non-empty batch")
        x = x.reshape(x.shape[0], -1)
        if x.shape[1] != self._num_features:
            raise ValueError("feature count does not match the schema")
        return x.to(device=self.device, dtype=self.x_dtype).reshape(-1, *self._shape)

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

    def _step(self, x: Tensor, y: Tensor, scale: float = 1.0) -> Tensor:
        self._model.train()
        self._projection.train()
        self._prototype_cache = None
        augmented = self._augment(x)
        if augmented.shape != x.shape:
            raise ValueError("augment must preserve the batch shape")
        views = torch.stack(
            [
                F.normalize(self._projection(self._encode(x)), dim=1),
                F.normalize(self._projection(self._encode(augmented)), dim=1),
            ],
            dim=1,
        )
        loss = self._loss(views, y) * scale
        if not torch.isfinite(loss):
            raise ValueError("SCR loss is not finite")
        self._optimiser.zero_grad()
        loss.backward()
        self._optimiser.step()
        return loss.detach()

    def batch_train(self, x: Tensor, y: Tensor) -> None:
        """Train using historical memory, then insert the incoming examples."""
        current_x = self._prepare_x(x)
        if y.shape != (x.shape[0],):
            raise ValueError("y must contain one class index per input row")
        y = y.to(device=self.device, dtype=self.y_dtype)
        if ((y < 0) | (y >= self._num_classes)).any():
            raise ValueError("class index is outside the schema")
        losses = []
        for _ in range(self._substeps):
            train_x, train_y = current_x, y
            if self._buffer.count:
                n = (
                    x.shape[0]
                    if self._replay_batch_size is None
                    else self._replay_batch_size
                )
                replay = self._buffer.sample(n)
                train_x = torch.cat((current_x, self._prepare_x(replay["x"])))
                train_y = torch.cat((y, replay["y"]))
            losses.append(self._step(train_x, train_y))
        self._buffer.update(x=x.detach().reshape(x.shape[0], -1), y=y)
        self._prototype_cache = None
        self.last_loss = torch.stack(losses).mean().item()

    @torch.no_grad()
    def _memory_features(self, x: Tensor) -> Tensor:
        was_training = self._model.training
        self._model.eval()
        try:
            return torch.cat(
                [
                    F.normalize(self._encode(self._prepare_x(batch)), dim=1)
                    for batch in x.split(self._feature_batch_size)
                ]
            )
        finally:
            self._model.train(was_training)

    @torch.no_grad()
    def batch_predict_proba(self, x: Tensor) -> Tensor:
        """Return NCM distance scores; empty memory gives uniform probabilities."""
        prepared = self._prepare_x(x)
        if not self._buffer.count:
            return torch.full(
                (x.shape[0], self._num_classes),
                1 / self._num_classes,
                device=self.device,
                dtype=self.x_dtype,
            )
        if self._prototype_cache is None:
            memory = self._buffer.array()
            features = self._memory_features(memory["x"])
            classes = memory["y"].unique(sorted=True)
            prototypes = torch.stack(
                [
                    F.normalize(features[memory["y"] == c].mean(0), dim=0)
                    for c in classes
                ]
            )
            self._prototype_cache = classes, prototypes
        classes, prototypes = self._prototype_cache
        query = self._memory_features(prepared)
        scores = torch.full(
            (x.shape[0], self._num_classes),
            -torch.inf,
            device=self.device,
            dtype=self.x_dtype,
        )
        scores[:, classes] = (
            -torch.cdist(query, prototypes).square() / self._ncm_temperature
        )
        return scores.softmax(1)

    def predict_proba(self, instance: Instance) -> LabelProbabilities:
        """Return single-instance scores as a CPU NumPy array."""
        return (
            self.batch_predict_proba(torch.as_tensor(instance.x).reshape(1, -1))
            .to(device="cpu", dtype=torch.float64)
            .numpy()
            .reshape(-1)
        )

    def attach_with(self, source: Dispatcher) -> "SCR":
        """Attach optional review to the existing task-end event, once per source."""
        if self._review and self._dispatcher is not source:
            source.subscribe(TrainTaskEnd, self._on_task_end)
            self._dispatcher = source
        return self

    def _on_task_end(self, event: TrainTaskEnd) -> None:
        if not self._buffer.count:
            return
        memory = self._buffer.array()
        permutation = torch.randperm(self._buffer.count, generator=self._buffer._rng)
        for indices in permutation.split(self._review_batch_size):
            self._step(
                self._prepare_x(memory["x"][indices]),
                memory["y"][indices],
                self._review_scale,
            )
        self.completed_reviews += 1

    def __str__(self) -> str:
        return f"SCR(buffer_capacity={self._buffer.capacity}, temperature={self._loss.temperature})"
