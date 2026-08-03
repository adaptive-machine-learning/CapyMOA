import math
import sys

import numpy as np

from capymoa.base import AnomalyDetector
from capymoa.instance import Instance
from capymoa.stream import Schema
from capymoa.type_alias import LabelIndex

__all__ = ["RSHash"]


class RSHashCountMinSketch:
    """Count-min sketch of ``w`` tables, each with a hash range of ``p``.

    :param p: Range of each hash function, i.e. cells per table.
    :param w: Number of pairwise independent hash tables.
    :param rng: Source of randomness for the hash keys.
    """

    def __init__(
        self, p: int = 10_000, w: int = 4, rng: np.random.Generator | None = None
    ):
        rng = rng if rng is not None else np.random.default_rng()
        self.p = p
        self.w = w
        self.hash_table = np.zeros((w, p), dtype=np.int32)
        self.hash_keys = [
            int(rng.integers(0, sys.maxsize, dtype=np.int64)) for _ in range(w)
        ]

    def _indices(self, x: np.ndarray) -> list[int]:
        payload = x.tobytes()
        return [hash((key, payload)) % self.p for key in self.hash_keys]

    def add(self, x: np.ndarray) -> None:
        for k, i in enumerate(self._indices(x)):
            self.hash_table[k, i] += 1

    def remove(self, x: np.ndarray) -> None:
        for k, i in enumerate(self._indices(x)):
            if self.hash_table[k, i] > 0:
                self.hash_table[k, i] -= 1

    def estimate(self, x: np.ndarray) -> int:
        return int(min(self.hash_table[k, i] for k, i in enumerate(self._indices(x))))


class RSHashComponent:
    """A single base detector of the RS-Hash ensemble.

    Each component draws its own locality parameter ``f``, shift vector
    ``alpha``, and subspace ``V``, then maintains counts of the discretised
    training points in its own count-min sketch. Steps 1 to 3 of Section II.

    :param s: Sample size the parameter ranges are derived from.
    :param d: Number of attributes.
    :param w: Number of hash tables in this component's sketch.
    :param p: Hash range of this component's sketch.
    :param rng: NumPy RNG, seeded per component.
    """

    def __init__(
        self,
        s: int,
        d: int,
        w: int,
        p: int,
        rng: np.random.Generator,
        active_dims: np.ndarray | None = None,
    ):
        self.d = d
        self.f = rng.uniform(1.0 / math.sqrt(s), 1.0 - 1.0 / math.sqrt(s))
        self.shift = rng.uniform(0.0, self.f, size=self.d)
        base = max(2.0, 1.0 / self.f)
        low = max(1, math.ceil(1.0 + 0.5 * math.log(s, base)))
        high = max(low, math.floor(math.log(s, base)))
        r = min(int(rng.integers(low, high + 1)), self.d)

        self.V = np.zeros(self.d, dtype=bool)
        self.V[rng.choice(self.d, size=r, replace=False)] = True
        self.V = self.V if active_dims is None else self.V & active_dims

        self.sketch = RSHashCountMinSketch(p=p, w=w, rng=rng)

    def _discretize(self, x: np.ndarray) -> np.ndarray:
        y = np.full(self.d, -1, dtype=np.int64)
        y[self.V] = np.floor((x[self.V] + self.shift[self.V]) / self.f)
        return y

    def add(self, x: np.ndarray) -> None:
        self.sketch.add(self._discretize(x))

    def remove(self, x: np.ndarray) -> None:
        self.sketch.remove(self._discretize(x))

    def score(self, x: np.ndarray) -> float:
        return math.log2(self.sketch.estimate(self._discretize(x)) + 1)


class RSHash(AnomalyDetector):
    """RS-Hash: subspace outlier detection in linear time with randomized hashing.

    The paper describes two streaming variants (Section III): a sliding window and
    time-decayed scores. This module implements the sliding window, which the paper
    notes is straightforward because the count-min sketch supports both insertion
    and deletion.

    Reference:
    Sathe, S. and Aggarwal, C. C. (2016). Subspace Outlier Detection in Linear
    Time with Randomized Hashing. IEEE ICDM, pp. 459-468.

    Example:
    >>> from capymoa.datasets import ElectricityTiny
    >>> from capymoa.anomaly import RSHash
    >>> from capymoa.evaluation import AnomalyDetectionEvaluator
    >>> stream = ElectricityTiny()
    >>> schema = stream.get_schema()
    >>> learner = RSHash(schema)
    >>> evaluator = AnomalyDetectionEvaluator(schema)
    >>> while stream.has_more_instances():
    ...     instance = stream.next_instance()
    ...     proba = learner.score_instance(instance)
    ...     evaluator.update(instance.y_index, proba)
    ...     learner.train(instance)
    >>> auc = evaluator.auc()
    >>> print(f"AUC: {auc:.2f}")
    AUC: 0.61
    """

    def __init__(
        self,
        schema: Schema,
        m: int = 300,
        s: int = 1000,
        w: int = 4,
        p: int = 10_000,
        seed: int = 42,
    ):
        """Construct an RS-Hash anomaly detector.

        :param schema: Schema of the stream.
        :param m: Number of ensemble components.
        :param s: Sliding window length.
        :param w: Number of hash tables per component.
        :param p: Hash range per component.
        :param seed: Random seed.
        """

        super().__init__(schema, random_seed=seed)
        self.m = m
        self.s = s
        self.w = w
        self.p = p
        self.d = schema.get_num_attributes()

        self.rng = np.random.default_rng(seed)

        self.components: list[RSHashComponent] = []
        self.window: list[Instance] = []

        self.min: np.ndarray = np.zeros(self.d, dtype=float)
        self.max: np.ndarray = np.zeros(self.d, dtype=float)

    def _normalize(self, x: np.ndarray) -> np.ndarray:
        range = np.where(self.active_dims, self.max - self.min, 1.0)
        return (x - self.min) / range

    def _initialize(self) -> None:
        """Build the ensemble once the window has filled, then load it."""
        stacked = np.stack([instance.x for instance in self.window])
        self.min = stacked.min(axis=0)
        self.max = stacked.max(axis=0)

        self.active_dims = self.max > self.min

        self.components = [
            RSHashComponent(
                s=self.s,
                d=self.d,
                w=self.w,
                p=self.p,
                active_dims=self.active_dims,
                rng=np.random.default_rng(
                    int(self.rng.integers(0, sys.maxsize, dtype=np.int64))
                ),
            )
            for _ in range(self.m)
        ]

        for instance in self.window:
            self._add(self._normalize(instance.x))

    def _add(self, x: np.ndarray) -> None:
        for component in self.components:
            component.add(x)

    def _remove(self, x: np.ndarray) -> None:
        for component in self.components:
            component.remove(x)

    def train(self, instance: Instance) -> None:
        self.window.append(instance)

        if not self.components:
            if len(self.window) >= self.s:
                self._initialize()
            return

        self._add(self._normalize(instance.x))
        if len(self.window) > self.s:
            self._remove(self._normalize(self.window.pop(0).x))

    def predict(self, instance: Instance) -> LabelIndex | None:
        raise NotImplementedError(
            "RSHash does not implement predict. Use score_instance instead."
        )

    def score_instance(self, instance: Instance) -> float:
        """Return the anomaly score for the given instance.

        Higher values indicate more anomalous instances.

        RS-Hash reports a normality score, so this method multiplies the
        ensemble average by -1 to stay consistent with the other detectors.
        Instances arriving before the window has filled score 0.0.

        :param instance: The instance to score.
        :return: The anomaly score.
        """
        if not self.components:
            return 0.0

        x = self._normalize(instance.x)
        total = sum(component.score(x) for component in self.components)
        return -total / self.m
