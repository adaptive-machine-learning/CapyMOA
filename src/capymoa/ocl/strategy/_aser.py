"""Adversarial Shapley Value Experience Replay."""

from collections.abc import Callable

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from capymoa.base import BatchClassifier
from capymoa.core import Instance, LabelProbabilities
from capymoa.ocl.util._replay import ReservoirSampler
from capymoa.stream import Schema


def _knn_shapley(
    evaluation: Tensor,
    evaluation_y: Tensor,
    candidates: Tensor,
    candidate_y: Tensor,
    k: int,
) -> Tensor:
    """Return the reference KNN-SV recurrence, with one row per evaluation point.

    The reference's farthest-neighbor value, indicator/N, is retained even when
    N < k. In that case it need not equal the Shapley value of fixed-1/k utility.
    """
    n = candidates.shape[0]
    if n == 0:
        raise ValueError("KNN-SV requires at least one candidate")
    # cdist avoids allocating evaluation_count * candidate_count * feature_dim.
    order = torch.cdist(
        evaluation, candidates, compute_mode="donot_use_mm_for_euclid_dist"
    ).argsort(dim=1)
    matches = (candidate_y[order] == evaluation_y[:, None]).to(evaluation.dtype)
    next_matches = F.pad(matches[:, 1:], (0, 1))
    ranks = torch.arange(1, n + 1, device=evaluation.device, dtype=evaluation.dtype)
    factors = ranks.clamp(max=k) / (k * ranks)
    factors[-1] = 1 / n
    sorted_values = ((matches - next_matches) * factors).flip(1).cumsum(1).flip(1)
    return torch.zeros_like(sorted_values).scatter(1, order, sorted_values)


class ASER(BatchClassifier):
    """Adversarial Shapley Value Experience Replay (ASER).

    ASER [#aser]_ uses KNN-Shapley values of latent features for both replay
    retrieval and memory replacement. This implementation follows the authors'
    released ASER retrieval and update policies, including balanced
    candidates drawn before the disjoint cooperative evaluation set, and the
    random minority-class threshold used in memory updates.

    ``asv`` selects ASER's maximum cooperative minus minimum adversarial score.
    ``asvm`` selects ASER-mu's difference of means and is the released code's
    default. ``neg_sv`` uses only negative summed adversarial scores. Warm-up
    retrieval is random until more than ``buffer_capacity`` examples have been
    observed. All draws are without replacement and capped at available data.
    Memory first fills in stream order; once full it uses Shapley replacement,
    rather than reservoir replacement. No task labels or boundaries are needed.

    Training retains the reference learner's preliminary current and replay
    forward/backward passes, whose gradients are discarded before the combined
    mean cross-entropy update. These passes can affect dropout and batch
    normalization. Memory is updated once, after all substeps, using the updated
    encoder. Input normalization belongs in the stream or supplied model.

    A private CPU generator replaces the reference's global NumPy/Torch random
    generators; seeds therefore do not reproduce its exact random sequence.
    Class caches are per learner. If a small memory leaves no cooperative
    evaluation points, retrieval falls back to random sampling. Empty update
    candidate sets leave memory unchanged. The reference's N < k recurrence is
    retained; choose k no larger than candidate sets for exact KNN-Shapley values.

    >>> from capymoa.ocl.strategy import ASER
    >>> schema = Schema.from_custom(
    ...     ["x0", "x1", "class"], "class", {"class": ["a", "b"]}
    ... )
    >>> model = nn.Sequential(nn.Linear(2, 8), nn.ReLU(), nn.Linear(8, 2))
    >>> learner = ASER(schema, model, torch.optim.SGD(model.parameters(), lr=0.1))
    >>> learner.batch_train(torch.eye(2), torch.tensor([0, 1]))
    >>> learner.batch_predict_proba(torch.ones(3, 2)).shape
    torch.Size([3, 2])

    .. [#aser] Shim, Dongsub, Zheda Mai, Jihwan Jeong, Scott Sanner,
        Hyunwoo Kim, and Jongseong Jang. "Online Class-Incremental Continual
        Learning with Adversarial Shapley Value." AAAI 35(11), 2021, 9630-9638.
        https://doi.org/10.1609/aaai.v35i11.17159
    """

    def __init__(
        self,
        schema: Schema,
        model: nn.Module,
        optimiser: torch.optim.Optimizer,
        *,
        feature_extractor: Callable[[Tensor], Tensor] | None = None,
        buffer_capacity: int = 200,
        replay_batch_size: int = 10,
        samples_per_class: int = 2,
        knn_k: int = 5,
        score_type: str = "asvm",
        substeps: int = 1,
        feature_batch_size: int = 64,
        seed: int = 0,
        device: torch.device | str = "cpu",
    ) -> None:
        """Construct an ASER learner.

        :param schema: Fixed classification schema, including input shape.
        :param model: Classifier returning one logit per schema class.
        :param optimiser: Optimizer owning the model's trainable parameters.
            Use SGD with lr=0.1 for the reference training configuration.
        :param feature_extractor: Callable using the model's parameters. Defaults
            to ``features``, then ``forward_features``, then a sequential view of
            the model's children with the last child removed, as in the reference.
            Supply this explicitly if the last fallback does not fit the model.
        :param buffer_capacity: Maximum number of examples and labels in memory.
        :param replay_batch_size: Maximum number of replay examples per step.
        :param samples_per_class: Maximum samples per class for balanced draws.
            Update candidates include up to this times all schema classes.
        :param knn_k: Number of neighbors for the reference KNN-SV recurrence.
        :param score_type: ``asv`` (ASER), ``asvm`` (ASER-mu), or ``neg_sv``.
        :param substeps: Training steps per batch; memory is updated only once.
        :param feature_batch_size: Chunk size for feature extraction.
        :param seed: Seed of this learner's private sampling generator.
        :param device: Device for the classifier and memory tensors.
        """
        super().__init__(schema, seed)
        if not schema.is_classification():
            raise ValueError("ASER requires a classification schema")
        for name, value in (
            ("buffer_capacity", buffer_capacity),
            ("replay_batch_size", replay_batch_size),
            ("samples_per_class", samples_per_class),
            ("knn_k", knn_k),
            ("substeps", substeps),
            ("feature_batch_size", feature_batch_size),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if score_type not in {"asv", "asvm", "neg_sv"}:
            raise ValueError("score_type must be asv, asvm or neg_sv")
        if feature_extractor is not None and not callable(feature_extractor):
            raise TypeError("feature_extractor must be callable")
        if not isinstance(optimiser, torch.optim.Optimizer):
            raise TypeError("optimiser must be a torch optimizer")
        self.device = torch.device(device)
        self._model = model.to(self.device)
        trainable = {id(p) for p in model.parameters() if p.requires_grad}
        owned = {id(p) for group in optimiser.param_groups for p in group["params"]}
        if not trainable <= owned or not owned <= {id(p) for p in model.parameters()}:
            raise ValueError("Optimizer must cover all trainable model parameters only")
        self._optimiser = optimiser
        self.x_dtype = next(model.parameters(), torch.empty(0)).dtype
        self._shape = tuple(schema.shape)
        self._num_features = schema.get_num_attributes()
        self._num_classes = schema.get_num_classes()
        if feature_extractor is None:
            feature_extractor = getattr(model, "features", None)
        if feature_extractor is None:
            feature_extractor = getattr(model, "forward_features", None)
        if feature_extractor is None:
            feature_extractor = nn.Sequential(*list(model.children())[:-1])
        self._feature_extractor = feature_extractor
        self._rng = torch.Generator().manual_seed(seed)
        # Reuse named storage and sequential filling, never Algorithm R replacement.
        self._buffer = ReservoirSampler(
            buffer_capacity,
            {"x": ((self._num_features,), self.x_dtype), "y": ((), self.y_dtype)},
            self._rng,
        ).to(self.device)
        self._class_indices: dict[int, set[int]] = {}
        self._seen = 0
        self._replay_batch_size = replay_batch_size
        self._samples_per_class = samples_per_class
        self._knn_k = knn_k
        self._score_type = score_type
        self._substeps = substeps
        self._feature_batch_size = feature_batch_size
        self.last_loss: float | None = None

    def _prepare_x(self, x: Tensor) -> Tensor:
        if x.ndim < 2 or x.shape[0] == 0:
            raise ValueError("x must contain a non-empty batch")
        flat = x.reshape(x.shape[0], -1)
        if flat.shape[1] != self._num_features:
            raise ValueError("feature count does not match the schema")
        return flat.to(device=self.device, dtype=self.x_dtype)

    @torch.no_grad()
    def _features(self, x: Tensor) -> Tensor:
        modes = {module: module.training for module in self._model.modules()}
        self._model.eval()
        try:
            chunks = [
                self._feature_extractor(part.reshape(-1, *self._shape)).reshape(
                    part.shape[0], -1
                )
                for part in x.split(self._feature_batch_size)
            ]
            return torch.cat(chunks)
        finally:
            for module, training in modes.items():
                module.training = training

    def _shapley(self, ex: Tensor, ey: Tensor, cx: Tensor, cy: Tensor) -> Tensor:
        features = self._features(torch.cat((ex, cx)))
        if features.dtype in (torch.float16, torch.bfloat16):
            features = features.float()
        return _knn_shapley(
            features[: len(ex)], ey, features[len(ex) :], cy, self._knn_k
        )

    def _random_indices(self, count: int, excluded: Tensor | None = None) -> Tensor:
        available = torch.ones(self._buffer.count, dtype=torch.bool)
        if excluded is not None:
            available[excluded.cpu()] = False
        indices = available.nonzero(as_tuple=True)[0]
        return indices[torch.randperm(len(indices), generator=self._rng)[:count]].to(
            self.device
        )

    def _balanced_indices(self, excluded: Tensor | None = None) -> Tensor:
        omit = set() if excluded is None else set(excluded.cpu().tolist())
        selected = []
        for members in self._class_indices.values():
            eligible = torch.tensor(sorted(members - omit), dtype=torch.long)
            order = torch.randperm(len(eligible), generator=self._rng)
            selected.append(eligible[order[: self._samples_per_class]])
        indices = torch.cat(selected) if selected else torch.empty(0, dtype=torch.long)
        return indices.to(self.device)

    def _retrieve(self, x: Tensor, y: Tensor) -> dict[str, Tensor]:
        memory = self._buffer.array()
        if self._seen <= self._buffer.capacity:
            chosen = self._random_indices(self._replay_batch_size)
        else:
            candidates = self._balanced_indices()
            adv = self._shapley(x, y, memory["x"][candidates], memory["y"][candidates])
            if self._score_type == "neg_sv":
                scores = -adv.sum(0)
            else:
                evaluation = self._balanced_indices(candidates)
                if len(evaluation) == 0:
                    chosen = self._random_indices(self._replay_batch_size)
                    return {key: value[chosen] for key, value in memory.items()}
                coop = self._shapley(
                    memory["x"][evaluation],
                    memory["y"][evaluation],
                    memory["x"][candidates],
                    memory["y"][candidates],
                )
                scores = (
                    coop.max(0).values - adv.min(0).values
                    if self._score_type == "asv"
                    else coop.mean(0) - adv.mean(0)
                )
            chosen = candidates[
                scores.argsort(descending=True)[: self._replay_batch_size]
            ]
        return {key: value[chosen] for key, value in memory.items()}

    def _write(self, indices: Tensor, x: Tensor, y: Tensor) -> None:
        memory = self._buffer.array()
        for index, label in zip(indices.cpu().tolist(), y.cpu().tolist()):
            old = int(memory["y"][index])
            self._class_indices[old].discard(index)
            self._class_indices.setdefault(label, set()).add(index)
        memory["x"][indices] = x.detach()
        memory["y"][indices] = y.detach()

    @torch.no_grad()
    def _update_memory(self, x: Tensor, y: Tensor) -> None:
        fill = min(self._buffer.capacity - self._buffer.count, len(x))
        start = self._buffer.count
        if fill:
            self._buffer.update(x=x[:fill], y=y[:fill])
            for index, label in enumerate(y[:fill].cpu().tolist(), start=start):
                self._class_indices.setdefault(label, set()).add(index)
        remaining_x, remaining_y = x[fill:], y[fill:]
        if self._buffer.count == self._buffer.capacity:
            memory = self._buffer.array()
            threshold = torch.rand((), generator=self._rng).item() / self._num_classes
            counts = torch.zeros(self._num_classes, device=self.device)
            for label, indices in self._class_indices.items():
                counts[label] = len(indices)
            minority = counts[remaining_y] / self._buffer.capacity < threshold
            evaluation = self._balanced_indices()
            ex = torch.cat((memory["x"][evaluation], remaining_x[minority]))
            ey = torch.cat((memory["y"][evaluation], remaining_y[minority]))
            candidates = self._random_indices(
                self._samples_per_class * self._num_classes, evaluation
            )
            n_old = len(candidates)
            if n_old and len(remaining_x):
                cx = torch.cat((memory["x"][candidates], remaining_x))
                cy = torch.cat((memory["y"][candidates], remaining_y))
                order = self._shapley(ex, ey, cx, cy).sum(0).argsort(descending=True)
                keep, discard = order[:n_old], order[n_old:]
                new = keep[keep >= n_old] - n_old
                old = candidates[discard[discard < n_old]]
                self._write(old, remaining_x[new], remaining_y[new])
        self._seen += len(x)

    def batch_train(self, x: Tensor, y: Tensor) -> None:
        """Train using reference ASER ordering, then update memory once."""
        x = self._prepare_x(x)
        y = y.to(device=self.device, dtype=self.y_dtype)
        if y.shape != (len(x),) or ((y < 0) | (y >= self._num_classes)).any():
            raise ValueError("y must contain one valid class index per example")
        self._model.train()
        losses = []
        for _ in range(self._substeps):
            self._optimiser.zero_grad()
            F.cross_entropy(self._model(x.reshape(-1, *self._shape)), y).backward()
            replay = self._retrieve(x, y)
            if len(replay["y"]):
                F.cross_entropy(
                    self._model(replay["x"].reshape(-1, *self._shape)), replay["y"]
                ).backward()
            self._optimiser.zero_grad()
            combined = torch.cat((replay["x"], x)).reshape(-1, *self._shape)
            loss = F.cross_entropy(self._model(combined), torch.cat((replay["y"], y)))
            loss.backward()
            self._optimiser.step()
            losses.append(loss.detach())
        self._update_memory(x, y)
        self.last_loss = torch.stack(losses).mean().item()

    @torch.no_grad()
    def batch_predict_proba(self, x: Tensor) -> Tensor:
        """Return class probabilities while preserving the model's modes."""
        modes = {module: module.training for module in self._model.modules()}
        self._model.eval()
        try:
            return self._model(self._prepare_x(x).reshape(-1, *self._shape)).softmax(
                dim=1
            )
        finally:
            for module, training in modes.items():
                module.training = training

    def predict_proba(self, instance: Instance) -> LabelProbabilities:
        """Return single-instance probabilities on CPU."""
        x = torch.as_tensor(instance.x).reshape(1, -1)
        return (
            self.batch_predict_proba(x)
            .to(device="cpu", dtype=torch.float64)
            .numpy()
            .reshape(-1)
        )

    def __str__(self) -> str:
        return f"ASER(score_type={self._score_type}, knn_k={self._knn_k}, buffer_capacity={self._buffer.capacity})"
