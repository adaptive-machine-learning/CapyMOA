import random
from collections import OrderedDict

import numpy as np

from capymoa.instance import Instance
from capymoa.stream._stream import Schema
from capymoa.base import AnomalyDetector

__all__ = ["RobustRandomCutForest"]


class RCLeaf:
    __slots__ = ["depth", "up", "mass", "bbox", "instance"]

    def __init__(self, instance: Instance, side, depth=None, up=None, mass=1):
        self.instance = instance
        self.depth = depth
        self.up = up
        self.mass = mass
        self.bbox = self.instance.x.reshape(1, -1)

    @property
    def x(self):
        return self.instance.x


class RCBranch:
    __slots__ = ["feature", "split_value", "left", "right", "up", "mass", "bbox"]

    ROOT = "root"
    LEFT = "left"
    RIGHT = "right"

    def __init__(
        self,
        feature,
        split_value,
        side,
        left=None,
        right=None,
        up=None,
        mass=0,
        bbox=None,
    ):
        self.feature = feature
        self.split_value = split_value
        self.left = left
        self.right = right
        self.up = up
        self.mass = mass
        self.bbox = bbox

    def __repr__(self):
        return "Branch(feature={}, split_value={:.2f})".format(
            self.feature, self.split_value
        )


class RCTree:
    def __init__(
        self, X, tree_size: int, num_attributes: int, random_state=None, precision=9
    ):
        if isinstance(random_state, int):
            self.rng = random.Random(random_state)
        elif isinstance(random_state, random.Random):
            self.rng = random_state
        else:
            self.rng = random.Random()

        self.root = None
        self.tree_size = tree_size
        self.num_attributes = num_attributes
        self.leaves = OrderedDict()
        if X is not None:
            self.root = self._mktree(X, parent=None, side=RCBranch.ROOT, depth=0)
            assert self.root is not None
            self.root.up = None
            self._count_all_top_down(self.root)
            self._get_bbox_top_down(self.root)

    def _get_min_max(self, X: list[Instance]):
        a = np.array([inst.x for inst in X])
        min_vals = np.min(a, axis=0)
        max_vals = np.max(a, axis=0)
        return min_vals, max_vals

    def _mktree(self, X, parent=None, side=RCBranch.ROOT, depth=0):
        if len(X) == 1:
            leaf = RCLeaf(instance=X[0], side=side, depth=depth, up=parent, mass=1)
            self.leaves[X[0]] = leaf
            return leaf

        min_values, max_values = self._get_min_max(X)

        weights = max_values - min_values
        if weights.sum() == 0:
            # all points are identical; build leaf
            leaf = RCLeaf(instance=X[0], side=side, depth=depth, up=parent, mass=len(X))
            for inst in X:
                self.leaves[inst] = leaf
            return leaf

        weights /= weights.sum()
        assert self.num_attributes is not None, "Tree dimensions not initialized"
        on = self.rng.choices(list(range(self.num_attributes)), weights=weights, k=1)[0]
        at = self.rng.uniform(min_values[on], max_values[on])
        X_l = [inst for inst in X if inst.x[on] < at]
        X_r = [inst for inst in X if inst.x[on] >= at]

        branch = RCBranch(feature=on, split_value=at, up=parent, side=side)
        left = self._mktree(X_l, parent=branch, side=RCBranch.LEFT, depth=depth + 1)
        right = self._mktree(X_r, parent=branch, side=RCBranch.RIGHT, depth=depth + 1)

        branch.left = left
        branch.right = right

        return branch

    def map_leaves(self, node, op=(lambda x: None), *args, **kwargs):
        if isinstance(node, RCBranch):
            if node.left:
                self.map_leaves(node.left, op=op, *args, **kwargs)
            if node.right:
                self.map_leaves(node.right, op=op, *args, **kwargs)
        else:
            op(node, *args, **kwargs)

    def forget_point(self, instance: Instance):
        try:
            leaf = self.leaves[instance]
        except KeyError:
            raise KeyError("Leaf must be a key to self.leaves")
        if leaf.mass > 1:
            self._update_leaf_count_upwards(leaf, inc=-1)
            return self.leaves.pop(instance)
        if leaf is self.root:
            self.root = None
            self.num_attributes = None
            return self.leaves.pop(instance)
        parent = leaf.up
        if leaf is parent.left:
            sibling = parent.right
        else:
            sibling = parent.left
        if parent is self.root:
            del parent
            sibling.up = None
            self.root = sibling
            if isinstance(sibling, RCLeaf):
                sibling.depth = 0
            else:
                self.map_leaves(sibling, op=self._increment_depth, inc=-1)
            return self.leaves.pop(instance)
        grandparent = parent.up
        sibling.up = grandparent
        if parent is grandparent.left:
            grandparent.left = sibling
        else:
            grandparent.right = sibling
        parent = grandparent
        self.map_leaves(sibling, op=self._increment_depth, inc=-1)
        self._update_leaf_count_upwards(parent, inc=-1)
        point = leaf.x
        self._relax_bbox_upwards(parent, point)
        return self.leaves.pop(instance)

    def _update_leaf_count_upwards(self, node, inc=1):
        while node:
            node.mass += inc
            node = node.up

    def insert_point(self, instance: Instance, tolerance=None):
        point = instance.x
        if not isinstance(point, np.ndarray):
            point = np.asarray(point)
        point = point.ravel()

        if self.root is None:
            leaf = RCLeaf(instance=instance, side=RCBranch.ROOT, depth=0)
            self.root = leaf
            self.num_attributes = point.size
            self.leaves[instance] = leaf
            return leaf
        try:
            assert point.size == self.num_attributes
        except ValueError:
            raise ValueError("Point must be same dimension as existing points in tree.")
        try:
            assert instance not in self.leaves
        except KeyError:
            raise KeyError("Index already exists in leaves dict.")
        duplicate = self.find_duplicate(instance, tolerance=tolerance)
        if duplicate:
            self._update_leaf_count_upwards(duplicate, inc=1)
            self.leaves[instance] = duplicate
            return duplicate
        node = self.root
        parent = node.up
        maxdepth = max([leaf.depth for leaf in self.leaves.values()])
        depth = 0
        branch = None
        leaf = None
        side = RCBranch.ROOT
        for _ in range(maxdepth + 1):
            assert node is not None
            assert node.bbox is not None
            bbox = node.bbox
            cut_dimension, cut = self._insert_point_cut(point, bbox)
            if cut <= bbox[0, cut_dimension]:
                leaf = RCLeaf(instance=instance, depth=depth, side=RCBranch.LEFT)
                branch = RCBranch(
                    feature=cut_dimension,
                    split_value=cut,
                    left=leaf,
                    right=node,
                    mass=(leaf.mass + node.mass),
                    side=side,
                )
                break
            elif cut >= bbox[-1, cut_dimension]:
                leaf = RCLeaf(instance=instance, depth=depth, side=RCBranch.RIGHT)
                branch = RCBranch(
                    feature=cut_dimension,
                    split_value=cut,
                    left=node,
                    right=leaf,
                    mass=(leaf.mass + node.mass),
                    side=side,
                )
                break
            else:
                depth += 1
                assert isinstance(node, RCBranch)
                if point[node.feature] <= node.split_value:
                    parent = node
                    node = node.left
                    side = RCBranch.LEFT
                else:
                    parent = node
                    node = node.right
                    side = RCBranch.RIGHT

        assert branch is not None, "Error with program logic: a cut was not found."
        assert node is not None
        assert leaf is not None
        node.up = branch
        leaf.up = branch
        branch.up = parent
        if parent is not None:
            assert side is not None
            setattr(parent, side, branch)
        else:
            self.root = branch
        self.map_leaves(branch, op=self._increment_depth, inc=1)
        self._update_leaf_count_upwards(parent, inc=1)
        self._tighten_bbox_upwards(branch)
        self.leaves[instance] = leaf
        return leaf

    def train(self, instance: Instance):
        if hasattr(self, "_train_index"):
            self._train_index += 1
        else:
            self._train_index = len(self.leaves)

        if len(self.leaves) >= self.tree_size:
            if self.leaves:
                oldest_inst = next(iter(self.leaves))
                self.forget_point(oldest_inst)

        self.insert_point(instance)

    def score_instance(self, instance: Instance) -> float:
        if not self.leaves:
            return 0.5

        temp_leaf = self.insert_point(instance)

        score = self.codisp(temp_leaf)

        self.forget_point(instance)

        return score

    def query(self, point, node=None):
        if not isinstance(point, np.ndarray):
            point = np.asarray(point)
        point = point.ravel()
        if node is None:
            node = self.root
        return self._query(point, node)

    def disp(self, leaf):
        if not isinstance(leaf, RCLeaf):
            try:
                leaf = self.leaves[leaf]
            except KeyError:
                raise KeyError("leaf must be a Leaf instance or key to self.leaves")
        if leaf is self.root:
            return 0

        parent = leaf.up
        assert parent is not None
        if leaf is parent.left:
            sibling = parent.right
        else:
            sibling = parent.left
        displacement = sibling.mass
        return displacement

    def codisp(self, leaf):
        if not isinstance(leaf, RCLeaf):
            try:
                leaf = self.leaves[leaf]
            except KeyError:
                raise KeyError("leaf must be a Leaf instance or key to self.leaves")

        if leaf is self.root:
            return 0
        node = leaf

        assert node.depth is not None
        results = []
        for _ in range(node.depth):
            parent = node.up
            if parent is None:
                break
            if node is parent.left:
                sibling = parent.right
            else:
                sibling = parent.left
            num_deleted = node.mass
            displacement = sibling.mass
            result = displacement / num_deleted
            results.append(result)
            node = parent
        co_displacement = max(results)
        return co_displacement

    def codisp_with_cut_dimension(self, leaf):
        if not isinstance(leaf, RCLeaf):
            try:
                leaf = self.leaves[leaf]
            except KeyError:
                raise KeyError("leaf must be a Leaf instance or key to self.leaves")

        if leaf is self.root:
            return 0
        node = leaf
        results = []
        cut_dimensions = []

        assert node.depth is not None
        for _ in range(node.depth):
            parent = node.up
            if parent is None:
                break
            if node is parent.left:
                sibling = parent.right
            else:
                sibling = parent.left
            num_deleted = node.mass
            displacement = sibling.mass
            result = displacement / num_deleted
            results.append(result)
            cut_dimensions.append(parent.feature)
            node = parent
        argmax = np.argmax(results)

        return results[argmax], cut_dimensions[argmax]

    def get_bbox(self, branch=None):
        if branch is None:
            branch = self.root

        assert self.num_attributes is not None, "Tree dimensions not initialized"
        mins = np.full(self.num_attributes, np.inf)
        maxes = np.full(self.num_attributes, -np.inf)
        self.map_leaves(branch, op=self._get_bbox, mins=mins, maxes=maxes)
        bbox = np.vstack([mins, maxes])
        return bbox

    def find_duplicate(self, instance, tolerance=None):
        point = instance.x
        nearest = self.query(point)
        if tolerance is None:
            if (nearest.x == point).all():
                return nearest
        else:
            if np.isclose(nearest.x, point, rtol=tolerance).all():
                return nearest
        return None

    def _lr_branch_bbox(self, node):
        bbox = np.vstack(
            [
                np.minimum(node.left.bbox[0, :], node.right.bbox[0, :]),
                np.maximum(node.left.bbox[-1, :], node.right.bbox[-1, :]),
            ]
        )
        return bbox

    def _get_bbox_top_down(self, node):
        if isinstance(node, RCBranch):
            if node.left:
                self._get_bbox_top_down(node.left)
            if node.right:
                self._get_bbox_top_down(node.right)
            bbox = self._lr_branch_bbox(node)
            node.bbox = bbox

    def _count_all_top_down(self, node):
        if isinstance(node, RCBranch):
            if node.left:
                self._count_all_top_down(node.left)
            if node.right:
                self._count_all_top_down(node.right)

            assert node.left is not None
            assert node.right is not None
            node.mass = node.left.mass + node.right.mass

    def _count_leaves(self, node):
        num_leaves = np.array(0, dtype=np.int64)
        self.map_leaves(node, op=self._accumulate, accumulator=num_leaves)
        num_leaves = num_leaves.item()
        return num_leaves

    def _query(self, point, node):
        if isinstance(node, RCLeaf):
            return node
        else:
            if point[node.feature] <= node.split_value:
                return self._query(point, node.left)
            else:
                return self._query(point, node.right)

    def _increment_depth(self, node, inc=1):
        node.depth += inc

    def _accumulate(self, node, accumulator):
        accumulator += node.mass

    def _get_nodes(self, node, stack):
        stack.append(node)

    def _get_bbox(self, instance, mins, maxes):
        lt = instance.x < mins
        gt = instance.x > maxes
        mins[lt] = instance.x[lt]
        maxes[gt] = instance.x[gt]

    def _tighten_bbox_upwards(self, node):
        bbox = self._lr_branch_bbox(node)
        node.bbox = bbox
        node = node.up
        while node:
            lt = bbox[0, :] < node.bbox[0, :]
            gt = bbox[-1, :] > node.bbox[-1, :]
            lt_any = lt.any()
            gt_any = gt.any()
            if lt_any or gt_any:
                if lt_any:
                    node.bbox[0, :][lt] = bbox[0, :][lt]
                if gt_any:
                    node.bbox[-1, :][gt] = bbox[-1, :][gt]
            else:
                break
            node = node.up

    def _relax_bbox_upwards(self, node, point):
        while node:
            bbox = self._lr_branch_bbox(node)
            if not ((node.bbox[0, :] == point) | (node.bbox[-1, :] == point)).any():
                break
            node.bbox[0, :] = bbox[0, :]
            node.bbox[-1, :] = bbox[-1, :]
            node = node.up

    def _insert_point_cut(self, point, bbox):
        bbox_hat = np.empty((2, bbox.shape[1]))
        bbox_hat[0, :] = np.minimum(bbox[0, :], point)
        bbox_hat[-1, :] = np.maximum(bbox[-1, :], point)
        b_span = bbox_hat[-1, :] - bbox_hat[0, :]
        b_range = b_span.sum()
        r = self.rng.uniform(0, b_range)
        span_sum = np.cumsum(b_span)
        cut_dimension = np.inf
        for j in range(len(span_sum)):
            if span_sum[j] >= r:
                cut_dimension = j
                break
        if not np.isfinite(cut_dimension):
            raise ValueError("Cut dimension is not finite.")
        cut_dimension = int(cut_dimension)
        cut = bbox_hat[0, cut_dimension] + span_sum[cut_dimension] - r
        return cut_dimension, cut


class RobustRandomCutForest(AnomalyDetector):
    """Robust Random Cut Forest.

    Robust Random Cut Forest (RRCF) [#f0]_ is an algorithm for anomaly detection in
    dynamic data streams. It maintains a random cut-based data structure (the forest)
    that acts as a compact sketch or synopsis of the input stream. Anomalies are defined
    non-parametrically in terms of the "externality" a new point imposes on the existing
    data—that is, how much the new point influences the structure of the forest.

    This implementation is adapted from https://klabum.github.io/rrcf/

    >>> from capymoa.datasets import ElectricityTiny
    >>> from capymoa.anomaly import RobustRandomCutForest
    >>> from capymoa.evaluation import AnomalyDetectionEvaluator

    >>> stream = ElectricityTiny()
    >>> schema = stream.get_schema()
    >>> learner = RobustRandomCutForest(schema, tree_size=256, n_trees=100, random_state=42)
    >>> evaluator = AnomalyDetectionEvaluator(schema)
    >>> while stream.has_more_instances():
    ...     instance = stream.next_instance()
    ...     proba = learner.score_instance(instance)
    ...     evaluator.update(instance.y_index, proba)
    ...     learner.train(instance)
    >>> auc = evaluator.auc()
    >>> print(f"AUC: {auc:.2f}")
    AUC: 0.56


    .. [#f0] Guha, S., Mishra, N., Roy, G., & Schrijvers, O. (2016, June). Robust random
        cut forest based anomaly detection on streams. In International conference on
        machine learning (pp. 2712-2721). PMLR.
    """

    def __init__(self, schema: Schema, tree_size=1000, n_trees=100, random_state=42):
        super().__init__(schema, random_state)
        self.tree_size = tree_size
        self.n_trees = n_trees
        if random_state is not None:
            self.rng = random.Random(random_state)
        else:
            self.rng = random.Random()
        self._trees: list[RCTree] = [
            RCTree(
                X=None,
                precision=9,
                random_state=self.rng,
                tree_size=tree_size,
                num_attributes=schema.get_num_attributes(),
            )
            for _ in range(n_trees)
        ]

    def train(self, instance: Instance):
        for tree in self._trees:
            tree.train(instance)

    def predict(self, instance: Instance) -> int | None:
        return super().predict(instance)

    def score_instance(self, instance: Instance) -> float:
        if len(self._trees[0].leaves) > 0:
            score = 0.0
            for tree in self._trees:
                score += tree.score_instance(instance) / self.n_trees
            return score
        else:
            return 0.5
