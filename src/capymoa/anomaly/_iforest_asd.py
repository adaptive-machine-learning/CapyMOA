from __future__ import annotations

import math
import random
import typing
from itertools import count

from capymoa.base import AnomalyDetector
from capymoa.instance import Instance
from capymoa.stream._stream import Schema
from capymoa.type_alias import LabelIndex

__all__ = ["IForestASD"]


class Leaf:
    def __init__(self, mass: int):
        self._mass = mass

    def walk(self, instance: Instance):
        yield self

    @property
    def mass(self):
        return self._mass


class Branch:
    def __init__(self, left, right, feature, split_value):
        self.children = [left, right]
        self.feature = feature
        self.split_value = split_value

    def walk(self, instance: Instance) -> typing.Iterable[typing.Union[Branch, Leaf]]:
        """Iterate over the nodes of the path induced by instance."""
        yield self
        yield from self.next(instance).walk(instance)

    @property
    def left(self) -> typing.Union[Branch, Leaf]:
        return self.children[0]

    @property
    def right(self) -> typing.Union[Branch, Leaf]:
        return self.children[1]

    @property
    def mass(self):
        return self.left.mass + self.right.mass

    def next(self, instance: Instance) -> typing.Union[Branch, Leaf]:
        try:
            value = instance.x[self.feature]
        except (KeyError, TypeError, IndexError) as e:
            raise ValueError(f"Cannot access feature {self.feature} in instance: {e}")

        if value < self.split_value:
            return self.left
        return self.right


def make_isolation_tree(
    X: list[Instance],
    *,
    height_limit,
    rng: random.Random,
    features,
):
    _attributes = features.copy()

    if height_limit == 0 or len(X) == 1:
        return Leaf(len(X))

    while len(_attributes) > 0:
        on = rng.choice(_attributes)
        a = float(min([inst.x[on] for inst in X]))
        b = float(max([inst.x[on] for inst in X]))
        if a != b:
            break
        _attributes.remove(on)
    else:
        return Leaf(len(X))

    at = rng.uniform(a, b)

    # Build the left node
    left = make_isolation_tree(
        [inst for inst in X if inst.x[on] < at],
        height_limit=height_limit - 1,
        rng=rng,
        features=features,
    )

    # Build the right node
    right = make_isolation_tree(
        [inst for inst in X if inst.x[on] >= at],
        height_limit=height_limit - 1,
        rng=rng,
        features=features,
    )

    branch = Branch(left, right, on, at)
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
            height_limit=height_limit,
            rng=rng,
            features=features,
        )

    def score_instance(self, instance: Instance) -> float:
        score = 0.0
        node = self._root
        for node in self._root.walk(instance):
            score += 1

        if node.mass > 1:
            score += c(node.mass)

        return score


class IForestASD(AnomalyDetector):
    """iForestASD

    Reference:
    `Ding, Z., & Fei, M. (2013). An anomaly detection approach based on isolation
    forest algorithm for streaming data using sliding window. IFAC proceedings
    volumes, 46(20), 12-17.`

    Example:
    >>> from capymoa.datasets import ElectricityTiny
    >>> from capymoa.anomaly import IForestASD
    >>> from capymoa.evaluation import AnomalyDetectionEvaluator
    >>> stream = ElectricityTiny()
    >>> schema = stream.get_schema()
    >>> learner = IForestASD(schema, window_size=256, n_trees=100,
    ...                      sample_size=64, random_state=42)
    >>> evaluator = AnomalyDetectionEvaluator(schema)
    >>> while stream.has_more_instances():
    ...     instance = stream.next_instance()
    ...     proba = learner.score_instance(instance)
    ...     evaluator.update(instance.y_index, proba)
    ...     learner.train(instance)
    >>> auc = evaluator.auc()
    >>> print(f"AUC: {auc:.2f}")
    AUC: 0.61

    TODO: implement concept drift method
    """

    def __init__(
        self,
        schema: Schema,
        window_size: int = 2048,
        n_trees: int = 100,
        sample_size: int = 256,
        height_limit: int | None = None,
        random_state: int | None = None,
    ):
        """Initialize the IForestASD anomaly detector.
        :param schema: The schema of the data stream.
        :param window_size: The size of the sliding window to maintain.
        :param n_trees: The number of isolation trees to build.
        :param sample_size: The number of instances to sample for each tree.
        :param height_limit: The maximum height of the isolation trees. If None, it will be set to ceil(log2(sample_size)).
        :param random_state: The seed for the random number generator.
        """
        super().__init__(
            schema=schema, random_seed=random_state if random_state is not None else 1
        )
        self.window_size = window_size
        self.n_trees = n_trees
        self.sample_size = sample_size
        self.height_limit = height_limit
        self.random_state = random_state
        self.instances = []
        self._trees: list[IsolationTree] = []
        self.id_counter = count(start=0)

    def score_instance(self, instance: Instance) -> float:
        """Calculate the anomaly score for an instance.

        A high score is indicative of an anomaly.

        :param instance: The instance to score.
        :return: The anomaly score between 0 and 1.
        """

        if len(self._trees) == 0:
            return 0.5

        score = 0.0
        for t in self._trees:
            score += t.score_instance(instance)

        score /= len(self._trees)
        score /= c(self.sample_size)
        score = 2**-score

        return score

    def train(self, instance: Instance):
        self.instances.append(instance)

        if len(self.instances) < self.window_size:
            return

        rng = random.Random(self.random_state)
        self._trees = []
        for _ in range(self.n_trees):
            sample = rng.sample(self.instances, self.sample_size)
            tree = IsolationTree(
                sample,
                features=list(range(self.schema.get_num_attributes())),
                height_limit=self.height_limit
                if self.height_limit is not None
                else math.ceil(math.log2(self.sample_size)),
                tree_id=next(self.id_counter),
                rng=rng,
            )
            self._trees.append(tree)

        self.instances = []

    def predict(self, instance) -> typing.Optional[LabelIndex]:
        """Predict is not implemented for anomaly detection.

        :param instance: The instance to predict.
        :raises NotImplementedError: This method is not applicable for anomaly detection.
        """
        raise NotImplementedError(
            "IForestASD does not implement predict. Use score_instance instead."
        )
