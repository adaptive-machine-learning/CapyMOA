from __future__ import annotations

import math
import random
import typing
from capymoa.base import AnomalyDetector
from capymoa.instance import LabeledInstance
from capymoa.stream._stream import Schema
from capymoa.type_alias import LabelIndex


__all__ = ["StreamingIsolationForest"]


class SiLeaf:
    def __init__(self, X: list[LabeledInstance], up, side):
        self.instances = list(X)
        self.up = up
        self.side = side

    def walk(self, x):
        yield self

    def __repr__(self):
        return str(f"{self.mass}@{self.depth}")

    @property
    def n_nodes(self):
        return 1

    @property
    def n_branches(self):
        return 0

    @property
    def n_leaves(self):
        return 1

    @property
    def height(self):
        return 1

    @property
    def mass(self):
        return len(self.instances)

    @property
    def depth(self):
        return 1 + self.up.depth if self.up is not None else 0

    def remove_instance(self, x):
        self.instances.remove(x)

    def insert_instance(self, x):
        self.instances.append(x)


class SiBranch:
    ROOT = "root"
    LEFT = "left"
    RIGHT = "right"

    def __init__(
        self,
        X: list[LabeledInstance],
        left,
        right,
        feature,
        split_value,
        up,
        side,
        f_min,
        f_max,
    ):
        self.children = [left, right]
        self.feature = feature
        self.split_value = split_value
        self.up = up
        self.side = side
        self.feature_min = f_min
        self.feature_max = f_max
        self.instances = list(X)

    @property
    def repr_split(self):
        return f"{self.feature} < {self.split_value:.5f}"

    def walk(self, instance: LabeledInstance) -> typing.Iterable[SiBranch | SiLeaf]:
        """Iterate over the nodes of the path induced by x."""
        yield self
        yield from self.next(instance).walk(instance)

    def traverse(self, instance: LabeledInstance) -> SiBranch | SiLeaf:
        """Return the leaf corresponding to the given input."""
        for node in self.walk(instance):
            pass
        return node

    @property
    def n_nodes(self):
        """Number of descendants, including thyself."""
        return 1 + sum(child.n_nodes for child in self.children)

    @property
    def n_branches(self):
        """Number of branches, including thyself."""
        return 1 + sum(child.n_branches for child in self.children)

    @property
    def n_leaves(self):
        """Number of leaves."""
        return sum(child.n_leaves for child in self.children)

    @property
    def height(self):
        """Distance to the deepest descendant."""
        return 1 + max(child.height for child in self.children)

    @property
    def left(self) -> SiBranch | SiLeaf:
        return self.children[0]

    @left.setter
    def left(self, value):
        self.children[0] = value

    @property
    def right(self) -> SiBranch | SiLeaf:
        return self.children[1]

    @right.setter
    def right(self, value):
        self.children[1] = value

    @property
    def mass(self):
        return self.left.mass + self.right.mass

    def next(self, instance: LabeledInstance) -> SiBranch | SiLeaf:
        left, right = self.children
        try:
            value = instance.x[self.feature]
        except KeyError:
            raise KeyError(
                f"Feature '{self.feature}' is missing in the input: {instance}"
            )
        except TypeError:
            raise TypeError(
                f"Invalid input type for feature '{self.feature}': {instance}"
            )

        if value < self.split_value:
            return left
        return right

    def __repr__(self):
        return str(f"{self.repr_split}@{self.depth}")

    @property
    def depth(self):
        return 1 + self.up.depth if self.up is not None else 0

    def compute_min_feature_value(self):
        return min([inst.x[self.feature] for inst in self.instances])

    def compute_max_feature_value(self):
        return max([inst.x[self.feature] for inst in self.instances])

    def remove_instance(self, instance: LabeledInstance):
        self.instances.remove(instance)

    def insert_instance(self, instance: LabeledInstance):
        self.instances.append(instance)


def H(i):
    return math.log(i) + 0.5772156649


def c(n):
    return 2 * H(n - 1) - (2 * (n - 1) / n)


class StreamingIsolationTree:
    def __init__(
        self,
        X: list[LabeledInstance],
        features,
        height_limit: int,
        rng: random.Random = random,
    ):
        self.rng = rng
        self.features = features
        self.k = len(X)  # Number of instances to keep in the tree
        self.instances = X[:]
        self.n = len(X)  # Number of instances seen so far
        self.height_limit = height_limit
        self._root = self._mk_tree(
            self.instances,
            height=self.height_limit,
            rng=self.rng,
            attributes=self.features,
            up=None,
            side=SiBranch.ROOT,
        )

    def _mk_tree(
        self,
        X: list[LabeledInstance],
        *,
        height,
        rng: random.Random = random,
        attributes=None,
        up=None,
        side=SiBranch.ROOT,
    ):
        _attributes = attributes.copy()

        if height == 0 or len(X) == 1:
            return SiLeaf(X, up=up, side=side)

        while len(_attributes) > 0:
            on = rng.choices(population=_attributes)[0]
            a = float(min([inst.x[on] for inst in X]))
            b = float(max([inst.x[on] for inst in X]))
            if a != b:
                break
            _attributes.remove(on)
        else:
            return SiLeaf(X, up=up, side=side)

        at = rng.uniform(a, b)

        # Build the left node
        left = self._mk_tree(
            [inst for inst in X if inst.x[on] < at],
            height=height - 1,
            rng=rng,
            attributes=attributes,
            up=up,
            side=SiBranch.LEFT,
        )

        # Build the right node
        right = self._mk_tree(
            [inst for inst in X if inst.x[on] >= at],
            height=height - 1,
            rng=rng,
            attributes=attributes,
            up=up,
            side=SiBranch.RIGHT,
        )

        branch = SiBranch(X, left, right, on, at, up, side, a, b)
        left.up = right.up = branch
        return branch

    def score_instance(self, instance: LabeledInstance) -> float:
        score = 0.0

        for node in self._root.walk(instance):
            score += 1

        if node.mass > 1:
            score += c(node.mass)

        return score

    def train(self, instance: LabeledInstance) -> StreamingIsolationTree:
        """
        Train the Streaming Isolation Tree with a new instance.

        This method implements reservoir sampling to decide whether to replace
        an existing instance in the tree with the new instance. If the new instance
        is added, the tree is updated accordingly.

        Parameters:
        instance (LabeledInstance): The new instance to be trained on.

        Returns:
        StreamingIsolationTree: The updated tree after training.
        """

        if self.n < self.k:
            i = self.n
        else:
            i = self.rng.randint(0, self.n)

        if i < self.k:
            instance_ = self.instances[i]
            self.instances[i] = instance
            self.remove_instance(instance_)
            self.insert_instance(instance)

        self.n += 1

        return self

    def insert_instance(self, instance: LabeledInstance):
        # traverse x to find the feature range not including x and then regrow the tree from there
        for node in self._root.walk(instance):
            if isinstance(node, SiBranch):
                node.insert_instance(instance)
                if (
                    node.compute_min_feature_value() < node.feature_min
                    or node.compute_max_feature_value() > node.feature_max
                ):
                    break
        else:
            # Leaf node
            node.insert_instance(instance)
            if node.depth == self.height_limit:
                # we are at bottom, just insert the new instance
                return

        # here we either find a node does not contain the new instance with its feature range,
        # or a leaf not being at bottom. We should regrow the subtree
        X = node.instances

        h = self.height_limit - node.depth
        if node.side == SiBranch.ROOT:
            # The parent is the root
            self._root = self._mk_tree(
                X,
                height=h,
                rng=self.rng,
                attributes=self.features,
                up=None,
                side=SiBranch.ROOT,
            )
        elif node.side == SiBranch.LEFT:
            node.up.left = self._mk_tree(
                X,
                height=h,
                rng=self.rng,
                attributes=self.features,
                up=node.up,
                side=SiBranch.LEFT,
            )
        else:  # parent.side == SiBranch.RIGHT
            node.up.right = self._mk_tree(
                X,
                height=h,
                rng=self.rng,
                attributes=self.features,
                up=node.up,
                side=SiBranch.RIGHT,
            )

    def remove_instance(self, instance: LabeledInstance):
        # traverse x to find the feature range change by removing x and then regrow the tree from there
        for node in self._root.walk(instance):
            if isinstance(node, SiBranch):
                node.remove_instance(instance)
                if (
                    node.compute_min_feature_value() > node.feature_min
                    or node.compute_max_feature_value() < node.feature_max
                ):
                    break

        else:
            # Leaf node
            node.remove_instance(instance)
            if node.mass > 0:
                return
            else:
                node = node.up

        X = node.instances

        h = self.height_limit - node.depth
        if node.side == SiBranch.ROOT:
            # The parent is the root
            self._root = self._mk_tree(
                X,
                height=h,
                rng=self.rng,
                attributes=self.features,
                up=None,
                side=SiBranch.ROOT,
            )
        elif node.side == SiBranch.LEFT:
            node.up.left = self._mk_tree(
                X,
                height=h,
                rng=self.rng,
                attributes=self.features,
                up=node.up,
                side=SiBranch.LEFT,
            )
        else:  # parent.side == SiBranch.RIGHT
            node.up.right = self._mk_tree(
                X,
                height=h,
                rng=self.rng,
                attributes=self.features,
                up=node.up,
                side=SiBranch.RIGHT,
            )


class StreamingIsolationForest(AnomalyDetector):
    """Streaming Isolation Forest anomaly detector.
    This detector constructs an ensemble of isolation trees incrementally
    in a streaming manner. Each tree employs reservoir sampling to
    maintain a fixed-size window of training instances. The anomaly score
    of an instance is calculated as the average path length across all
    trees, normalized by the expected path length for a randomly chosen
    instance in a tree of equivalent size. Scores are scaled between
    0 and 1, with higher values indicating greater anomaly likelihood.

    Reference:
    Liu, J.J., Cassales, G.W., Liu, F.T., Pfahringer, B., Bifet, A. (2025).
    Streaming Isolation Forest. In: Wu, X., et al. Advances in Knowledge
    Discovery and Data Mining . PAKDD 2025. Lecture Notes in Computer Science(),
    vol 15870. Springer, Singapore. https://doi.org/10.1007/978-981-96-8170-9_8

    Example:
    >>> from capymoa.datasets import ElectricityTiny
    >>> from capymoa.anomaly import StreamingIsolationForest
    >>> from capymoa.evaluation import AnomalyDetectionEvaluator
    >>> stream = ElectricityTiny()
    >>> schema = stream.get_schema()
    >>> learner = StreamingIsolationForest(schema, window_size=256, n_trees=100, seed=42)
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
        window_size=256,
        n_trees=100,
        height=None,
        seed: int | None = None,
    ):
        """Construct a Streaming Isolation Forest anomaly detector.
        :param schema: The schema of the stream. If not provided, it will be inferred from the data.
        :param window_size: The size of the window for each tree.
        :param n_trees: The number of trees in the ensemble.
        :param height: The maximum height of each tree. If None, it will be set to log2(window_size).
        :param seed: Random seed for reproducibility.
        """

        super().__init__(schema=schema, random_seed=seed)
        self.n_trees = n_trees
        self._trees = []
        self.window_size = window_size
        self.height_limit = height or math.ceil(math.log2(window_size))
        self.instances = []
        self.rng = random.Random(self.random_seed)

    def train(self, instance: LabeledInstance):
        if len(self._trees) == 0:
            self.instances.append(instance)
            if len(self.instances) == self.window_size:
                for _ in range(self.n_trees):
                    t = StreamingIsolationTree(
                        self.instances,
                        list(range(self.schema.get_num_attributes())),
                        self.height_limit,
                        self.rng,
                    )
                    self._trees.append(t)
                self.instances = []
        else:
            for t in self._trees:
                t.train(instance)

    def score_instance(self, instance: LabeledInstance) -> float:
        if len(self._trees) == 0:
            return 0.5

        score = 0.0
        for t in self._trees:
            score += t.score_instance(instance)

        score /= len(self._trees)
        score /= c(self.window_size)
        score = 2**-score

        return score

    def predict(self, instance: LabeledInstance) -> typing.Optional[LabelIndex]:
        raise NotImplementedError(
            "StreamingIsolationForest does not implement predict. Use score_instance instead."
        )
