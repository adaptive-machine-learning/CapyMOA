from __future__ import annotations

import math
import random
import typing
from itertools import count

from capymoa.base import AnomalyDetector
from capymoa.instance import Instance
from capymoa.stream import Schema
from capymoa.type_alias import LabelIndex

__all__ = ["AdaptiveIsolationForest"]


class AIFLeaf:
    def __init__(self, X: list[Instance], up, side):
        self.instances = list(X)
        self.up = up
        self.side = side

    def walk(self, instance: Instance):
        yield self

    def __repr__(self):
        return str(f"{self.mass}@{self.depth}")

    @property
    def n_nodes(self):
        return 1

    @property
    def mass(self):
        return len(self.instances)

    @property
    def depth(self):
        return 1 + self.up.depth if self.up is not None else 0


class AIFBranch:
    ROOT = "root"
    LEFT = "left"
    RIGHT = "right"

    def __init__(self, X: list[Instance], left, right, feature, split_value, up, side):
        self.children = [left, right]
        self.feature = feature
        self.split_value = split_value
        self.up = up
        self.side = side
        self.instances = list(X)

    @property
    def repr_split(self):
        return f"{self.feature} < {self.split_value:.5f}"

    def walk(
        self, instance: Instance
    ) -> typing.Iterable[typing.Union[AIFBranch, AIFLeaf]]:
        """Iterate over the nodes of the path induced by instance."""
        yield self
        yield from self.next(instance).walk(instance)

    @property
    def n_nodes(self):
        """Number of descendants, including thyself."""
        return 1 + sum(child.n_nodes for child in self.children)

    @property
    def left(self) -> typing.Union[AIFBranch, AIFLeaf]:
        return self.children[0]

    @left.setter
    def left(self, value):
        self.children[0] = value

    @property
    def right(self) -> typing.Union[AIFBranch, AIFLeaf]:
        return self.children[1]

    @right.setter
    def right(self, value):
        self.children[1] = value

    @property
    def mass(self):
        return self.left.mass + self.right.mass

    def next(self, instance: Instance) -> typing.Union[AIFBranch, AIFLeaf]:
        try:
            value = instance.x[self.feature]
        except (KeyError, TypeError, IndexError) as e:
            raise ValueError(f"Cannot access feature {self.feature} in instance: {e}")

        if value < self.split_value:
            return self.left
        return self.right

    def __repr__(self):
        return str(f"{self.repr_split}@{self.depth}")

    @property
    def depth(self):
        return 1 + self.up.depth if self.up is not None else 0


def make_isolation_tree(
    X: list[Instance],
    *,
    height,
    rng: random.Random,
    attributes,
    up=None,
    side=AIFBranch.ROOT,
):
    _attributes = attributes.copy()

    if height == 0 or len(X) == 1:
        return AIFLeaf(X, up=up, side=side)

    while len(_attributes) > 0:
        on = rng.choices(population=_attributes)[0]
        a = float(min([inst.x[on] for inst in X]))
        b = float(max([inst.x[on] for inst in X]))
        if a != b:
            break
        _attributes.remove(on)
    else:
        return AIFLeaf(X, up=up, side=side)

    at = rng.uniform(a, b)

    # Build the left node
    left = make_isolation_tree(
        [inst for inst in X if inst.x[on] < at],
        height=height - 1,
        rng=rng,
        attributes=attributes,
        up=up,
        side=AIFBranch.LEFT,
    )

    # Build the right node
    right = make_isolation_tree(
        [inst for inst in X if inst.x[on] >= at],
        height=height - 1,
        rng=rng,
        attributes=attributes,
        up=up,
        side=AIFBranch.RIGHT,
    )

    branch = AIFBranch(X, left, right, on, at, up, side)
    left.up = right.up = branch
    return branch


def H(i):
    return math.log(i) + 0.5772156649


def c(n):
    return 2 * H(n - 1) - (2 * (n - 1) / n)


class IsolationTree:
    def __init__(
        self,
        X: list[Instance],
        features,
        height_limit: int,
        tree_id: int,
        rng: random.Random,
    ):
        self.id = tree_id
        self.features = features
        self.height_limit = height_limit
        self._root = make_isolation_tree(
            X,
            height=height_limit,
            rng=rng,
            attributes=features,
            up=None,
            side=AIFBranch.ROOT,
        )

    def score_instance(self, instance: Instance) -> float:
        score = 0.0
        node = self._root
        for node in self._root.walk(instance):
            score += 1

        if node.mass > 1:
            score += c(node.mass)

        return score

    def _get_all_leaves(self, node=None) -> list[AIFLeaf]:
        """Recursively collect all leaf nodes."""
        if node is None:
            node = self._root

        if isinstance(node, AIFLeaf):
            return [node]
        elif isinstance(node, AIFBranch):
            leaves = []
            leaves.extend(self._get_all_leaves(node.left))
            leaves.extend(self._get_all_leaves(node.right))
            return leaves
        return []

    @property
    def max_mass(self):
        leaves = self._get_all_leaves()
        return max([leaf.mass for leaf in leaves]) if leaves else 0

    @property
    def n_nodes(self):
        return self._root.n_nodes


class AdaptiveIsolationForest(AnomalyDetector):
    """Adaptive Isolation Forest for anomaly detection.

    This implementation adapts the Isolation Forest algorithm to be more adaptive
    in changing environments. It uses a sliding window approach and can replace
    trees based on their quality scores, combining tree size and maximum mass
    metrics to determine which trees to keep.

    References:

    Liu, J.J., Cassales, G.W., Liu, F.T., Pfahringer, B., Bifet, A. (2025).
    Adaptive Isolation Forest. In: Džeroski, S., Levatić, J., Pio, G.,
    Simidjievski, N. (eds) Discovery Science. DS 2025. Lecture Notes in
    Computer Science(), vol 16090. Springer, Cham.
    https://doi.org/10.1007/978-3-032-05461-6_24

    Example:

    >>> from capymoa.datasets import ElectricityTiny
    >>> from capymoa.anomaly import AdaptiveIsolationForest
    >>> from capymoa.evaluation import AnomalyDetectionEvaluator
    >>> stream = ElectricityTiny()
    >>> schema = stream.get_schema()
    >>> learner = AdaptiveIsolationForest(schema, window_size=256, n_trees=100)
    >>> evaluator = AnomalyDetectionEvaluator(schema)
    >>> while stream.has_more_instances():
    ...     instance = stream.next_instance()
    ...     proba = learner.score_instance(instance)
    ...     evaluator.update(instance.y_index, proba)
    ...     learner.train(instance)
    >>> auc = evaluator.auc()
    >>> print(f"AUC: {auc:.2f}")
    AUC: 0.81
    """

    def __init__(
        self,
        schema: Schema,
        window_size=256,
        n_trees=100,
        height=None,
        seed: int | None = None,
        m_trees=10,
        weights=0.5,
    ):
        """Construct an Adaptive Isolation Forest anomaly detector.

        :param schema: The schema of the stream.
        :param window_size: Size of the sliding window for training trees.
        :param n_trees: Number of trees in the ensemble.
        :param height: Maximum height of trees. If None, calculated as ceil(log2(window_size)).
        :param seed: Random seed for reproducibility.
        :param m_trees: Number of candidate trees to generate when replacing.
        :param weights: Weight for combining tree size and max mass scores (0-1).
        :param skip_default_strategy: If True, skip default replacement when quality score doesn't improve.
        """
        super().__init__(schema=schema, random_seed=seed if seed is not None else 1)
        self.n_trees = n_trees
        self._trees: list[IsolationTree] = []
        self.height_limit = height or math.ceil(math.log2(window_size))
        self.window_size = window_size
        self.instances: list[Instance] = []
        self.rng = random.Random(self.random_seed)
        self.id_counter = count(start=0)
        self.m_trees = m_trees
        self.weights = weights

    def _compute_tree_scores(self, trees: list[IsolationTree]) -> list[float]:
        """Compute quality scores for trees based on size and max mass."""
        if not trees:
            return []

        # Get metrics
        tree_sizes = [t.n_nodes for t in trees]
        max_masses = [t.max_mass for t in trees]

        # Normalize tree sizes (inverse - smaller is better)
        min_size, max_size = min(tree_sizes), max(tree_sizes)
        if max_size == min_size:
            norm_sizes = [0.0] * len(trees)
        else:
            norm_sizes = [(max_size - s) / (max_size - min_size) for s in tree_sizes]

        # Normalize max masses (larger is better)
        min_mass, max_mass = min(max_masses), max(max_masses)
        if max_mass == min_mass:
            norm_masses = [0.0] * len(trees)
        else:
            norm_masses = [(m - min_mass) / (max_mass - min_mass) for m in max_masses]

        # Combine scores
        scores = [
            self.weights * norm_sizes[i] + (1 - self.weights) * norm_masses[i]
            for i in range(len(trees))
        ]
        return scores

    def train(self, instance):
        """Train the model on a single instance.

        :param instance: The instance to train on.
        """
        self.instances.append(instance)

        if len(self.instances) == self.window_size:
            features = list(range(self.schema.get_num_attributes()))

            if len(self._trees) == 0:
                # Initialize trees
                while len(self._trees) < self.n_trees:
                    t = IsolationTree(
                        self.instances,
                        features,
                        self.height_limit,
                        next(self.id_counter),
                        self.rng,
                    )
                    self._trees.append(t)
            else:
                # Generate candidate trees
                candidates = [
                    IsolationTree(
                        self.instances,
                        features,
                        self.height_limit,
                        next(self.id_counter),
                        self.rng,
                    )
                    for _ in range(self.m_trees)
                ]

                # Evaluate all trees
                all_trees = candidates + self._trees
                all_scores = self._compute_tree_scores(all_trees)

                candidate_scores = all_scores[: self.m_trees]
                tree_scores = all_scores[self.m_trees :]

                # Find best candidate and worst existing tree
                best_candidate_idx = candidate_scores.index(max(candidate_scores))
                worst_tree_idx = tree_scores.index(min(tree_scores))

                # Replace if candidate is better
                if candidate_scores[best_candidate_idx] > tree_scores[worst_tree_idx]:
                    del self._trees[worst_tree_idx]
                else:  # or replace the oldest tree
                    del self._trees[0]

                self._trees.append(candidates[best_candidate_idx])
            self.instances = []

    def score_instance(self, instance: Instance) -> float:
        """Calculate the anomaly score for an instance.

        A high score is indicative of an anomaly.

        :param instance: The instance to score (must be a LabeledInstance).
        :return: The anomaly score between 0 and 1.
        """

        if len(self._trees) == 0:
            return 0.5

        score = 0.0
        for t in self._trees:
            score += t.score_instance(instance)

        score /= len(self._trees)
        score /= c(self.window_size)
        score = 2**-score

        return score

    def predict(self, instance) -> typing.Optional[LabelIndex]:
        """Predict is not implemented for anomaly detection.

        :param instance: The instance to predict.
        :raises NotImplementedError: This method is not applicable for anomaly detection.
        """
        raise NotImplementedError(
            "AdaptiveIsolationForest does not implement predict. Use score_instance instead."
        )
