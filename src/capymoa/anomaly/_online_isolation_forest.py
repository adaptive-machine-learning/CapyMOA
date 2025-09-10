from __future__ import annotations
from capymoa.base import AnomalyDetector
from capymoa.instance import Instance, LabelIndex
from capymoa.stream._stream import Schema
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from itertools import repeat
from multiprocessing import cpu_count
from numpy import (
    argsort,
    asarray,
    empty,
    finfo,
    inf,
    log,
    ndarray,
    sort,
    split,
    vstack,
    zeros,
)
from numpy.linalg import norm
from numpy.random import default_rng, Generator
from typing import Callable, Literal, Optional, Tuple


class OnlineIsolationForest(AnomalyDetector):
    """Online Isolation Forest

    This class implements the Online Isolation Forest (oIFOR) algorithm, which is
    an ensemble anomaly detector capable of adapting to concept drift.

    Reference:

    `Online Isolation Forest.
    Filippo Leveni, Guilherme Weigert Cassales, Bernhard Pfahringer, Albert Bifet, and Giacomo Boracchi.
    International Conference on Machine Learning (ICML), Proceedings of Machine Learning Research (PMLR), 2024.
    <https://proceedings.mlr.press/v235/leveni24a.html>`_

    Example:

    >>> from capymoa.datasets import ElectricityTiny
    >>> from capymoa.anomaly import OnlineIsolationForest
    >>> from capymoa.evaluation import AnomalyDetectionEvaluator
    >>> stream = ElectricityTiny()
    >>> schema = stream.get_schema()
    >>> learner = OnlineIsolationForest(schema=schema)
    >>> evaluator = AnomalyDetectionEvaluator(schema)
    >>> while stream.has_more_instances():
    ...     instance = stream.next_instance()
    ...     proba = learner.score_instance(instance)
    ...     evaluator.update(instance.y_index, proba)
    ...     learner.train(instance)
    >>> auc = evaluator.auc()
    >>> print(f"AUC: {auc:.2f}")
    AUC: 0.39

    """

    def __init__(
        self,
        schema: Optional[Schema] = None,
        random_seed: int = 1,
        num_trees: int = 32,
        max_leaf_samples: int = 32,
        growth_criterion: Literal["fixed", "adaptive"] = "adaptive",
        subsample: float = 1.0,
        window_size: int = 2048,
        branching_factor: int = 2,
        split: Literal["axisparallel"] = "axisparallel",
        n_jobs: int = 1,
    ):
        """Construct an Online Isolation Forest anomaly detector

        :param schema: The schema of the stream. If not provided, it will be inferred from the data.
        :param random_seed: Random seed for reproducibility.
        :param num_trees: Number of trees in the ensemble.
        :param window_size: The size of the window for each tree.
        :param branching_factor: Branching factor of each tree.
        :param max_leaf_samples: Maximum number of samples per leaf. When this number is reached, a split is performed.
        :param growth_criterion: When to perform a split. If 'adaptive', the max_leaf_samples grows with tree depth,
                                 otherwise 'fixed'.
        :param subsample: Probability of learning a new sample in each tree.
        :param split: Type of split performed at each node. Currently only 'axisparallel' is supported, which is the
                      same type used by the IsolationForest algorithm.
        :param n_jobs: Number of parallel jobs.
        """
        super().__init__(schema=schema, random_seed=random_seed)
        self.random_generator: Generator = default_rng(seed=self.random_seed)
        self.num_trees: int = num_trees
        self.window_size: int = window_size
        self.branching_factor: int = branching_factor
        self.max_leaf_samples: int = max_leaf_samples
        self.growth_criterion: Literal["fixed", "adaptive"] = growth_criterion
        self.subsample: float = subsample
        self.trees: list[OnlineIsolationTree] = []
        self.data_window: list[ndarray] = []
        self.data_size: int = 0
        self.normalization_factor: float = 0
        self.split: Literal["axisparallel"] = split
        self.n_jobs: int = cpu_count() if n_jobs == -1 else min(n_jobs, cpu_count())
        self.trees: list[OnlineIsolationTree] = [
            OnlineIsolationTree(
                max_leaf_samples=max_leaf_samples,
                growth_criterion=growth_criterion,
                subsample=self.subsample,
                branching_factor=self.branching_factor,
                data_size=self.data_size,
                split=self.split,
                random_seed=self.random_generator.integers(0, 2**32),
            )
            for _ in range(self.num_trees)
        ]

    def train(self, instance: Instance):
        data: ndarray = instance.x.reshape((1, -1))
        self._learn_batch(data)
        return

    def __str__(self):
        return "Online Isolation Forest"

    def predict(self, instance: Instance) -> Optional[LabelIndex]:
        pass

    def score_instance(self, instance: Instance) -> float:
        data: ndarray = instance.x.reshape((1, -1))
        return self._score_batch(data)[0]

    def _learn_batch(self, data: ndarray) -> OnlineIsolationForest:
        # Update the counter of data seen so far
        self.data_size += data.shape[0]
        # Compute the normalization factor
        self.normalization_factor: float = OnlineIsolationTree._get_random_path_length(
            self.branching_factor,
            self.max_leaf_samples,
            self.data_size * self.subsample,
        )
        # Instantiate a list of OnlineIsolationTrees' learn functions
        learn_funcs: list[Callable[[ndarray], OnlineIsolationForest]] = [
            tree._learn for tree in self.trees
        ]
        # OnlineIsolationTrees learn new data
        with ThreadPoolExecutor(max_workers=self.n_jobs) as executor:
            self.trees: list[OnlineIsolationTree] = list(
                executor.map(
                    lambda f, x: f(x), learn_funcs, repeat(data, self.num_trees)
                )
            )
        # If the window size is not None, add new data to the window and eventually remove old ones
        if self.window_size:
            # Update the window of data seen so far
            self.data_window += list(data)
            # If the window size is smaller than the number of data seen so far, unlearn old data
            if self.data_size > self.window_size:
                # Extract old data and update the window of data seen so far
                data, self.data_window = (
                    asarray(self.data_window[: self.data_size - self.window_size]),
                    self.data_window[self.data_size - self.window_size :],
                )
                # Update the counter of data seen so far
                self.data_size -= self.data_size - self.window_size
                # Compute the normalization factor
                self.normalization_factor: float = (
                    OnlineIsolationTree._get_random_path_length(
                        self.branching_factor,
                        self.max_leaf_samples,
                        self.data_size * self.subsample,
                    )
                )
                # Instantiate a list of OnlineIsolationTrees' unlearn functions
                unlearn_funcs: list[Callable[[ndarray], OnlineIsolationForest]] = [
                    tree._unlearn for tree in self.trees
                ]
                # OnlineIsolationTrees unlearn new data
                with ThreadPoolExecutor(max_workers=self.n_jobs) as executor:
                    self.trees: list[OnlineIsolationTree] = list(
                        executor.map(
                            lambda f, x: f(x),
                            unlearn_funcs,
                            repeat(data, self.num_trees),
                        )
                    )
        return self

    def _score_batch(self, data: ndarray) -> ndarray[float]:
        # Collect OnlineIsolationTrees' predict functions
        predict_funcs: list[Callable[[ndarray], OnlineIsolationForest]] = [
            tree._predict for tree in self.trees
        ]
        # Compute the depths of all samples in each tree
        with ThreadPoolExecutor(max_workers=self.n_jobs) as executor:
            depths: ndarray[float] = asarray(
                list(
                    executor.map(
                        lambda f, x: f(x), predict_funcs, repeat(data, self.num_trees)
                    )
                )
            ).T
        # Compute the mean depth of each sample along all trees
        mean_depths: ndarray[float] = depths.mean(axis=1)
        # Compute normalized mean depths
        normalized_mean_depths: ndarray[float] = 2 ** (
            -mean_depths / (self.normalization_factor + finfo(float).eps)
        )
        return normalized_mean_depths


class OnlineIsolationTree:
    def __init__(
        self,
        random_seed: int,
        max_leaf_samples: int,
        growth_criterion: Literal["fixed", "adaptive"],
        subsample: float,
        branching_factor: int,
        data_size: int,
        split: Literal["axisparallel"] = "axisparallel",
    ):
        self.random_generator: Generator = default_rng(seed=random_seed)
        self.max_leaf_samples: int = max_leaf_samples
        self.growth_criterion: Literal["fixed", "adaptive"] = growth_criterion
        self.subsample: float = subsample
        self.branching_factor: int = branching_factor
        self.data_size: int = data_size
        self.split: Literal["axisparallel"] = split
        self.depth_limit: float = OnlineIsolationTree._get_random_path_length(
            self.branching_factor,
            self.max_leaf_samples,
            self.data_size * self.subsample,
        )
        self.root: Optional[OnlineIsolationNode] = None
        self.next_node_index: int = 0

    @staticmethod
    def _get_random_path_length(
        branching_factor: int, max_leaf_samples: int, num_samples: float
    ) -> float:
        if num_samples < max_leaf_samples:
            return 0
        else:
            return log(num_samples / max_leaf_samples) / log(2 * branching_factor)

    @staticmethod
    def _get_multiplier(
        growth_criterion: Literal["fixed", "adaptive"], depth: int
    ) -> int:
        # Compute the multiplier according to the growth criterion
        if growth_criterion == "fixed":
            return 1
        elif growth_criterion == "adaptive":
            return 2**depth
        else:
            raise ValueError("Bad grow criterion {}".format(growth_criterion))

    @staticmethod
    def _split_data(
        data: ndarray, projection_vector: ndarray[float], split_values: ndarray[float]
    ) -> list[ndarray[int]]:
        # Project data using projection vector
        projected_data: ndarray = data @ projection_vector
        # Sort projected data and keep sort indices
        sort_indices: ndarray = argsort(projected_data)
        # Split data according to their membership
        partition: list[ndarray[int]] = split(
            sort_indices, projected_data[sort_indices].searchsorted(split_values)
        )
        return partition

    def _learn(self, data: ndarray) -> OnlineIsolationTree:
        # Subsample data in order to improve diversity among trees
        data: ndarray = data[
            self.random_generator.random(data.shape[0]) < self.subsample
        ]
        if data.shape[0] >= 1:
            # Update the counter of data seen so far
            self.data_size += data.shape[0]
            # Adjust depth limit according to data seen so far and branching factor
            self.depth_limit: float = OnlineIsolationTree._get_random_path_length(
                self.branching_factor, self.max_leaf_samples, self.data_size
            )
            # Recursively update the tree
            self.next_node_index, self.root = (
                self._recursive_build(data)
                if self.root is None
                else self._recursive_learn(self.root, data, self.next_node_index)
            )
        return self

    def _recursive_learn(
        self, node: OnlineIsolationNode, data: ndarray, node_index: int
    ) -> Tuple[int, OnlineIsolationNode]:
        # Update the number of data seen so far by the current node
        node.data_size += data.shape[0]
        # Update the vectors of minimum and maximum values seen so far by the current node
        node.min_values: ndarray = vstack([data, node.min_values]).min(axis=0)
        node.max_values: ndarray = vstack([data, node.max_values]).max(axis=0)
        # If the current node is a leaf, try to split it
        if node.children is None:
            # If there are enough samples to be split according to the max leaf samples and the depth limit has not been
            # reached yet, split the node
            if (
                node.data_size
                >= self.max_leaf_samples
                * OnlineIsolationTree._get_multiplier(self.growth_criterion, node.depth)
                and node.depth < self.depth_limit
            ):
                # Sample data_size points uniformly at random within the bounding box defined by the vectors of minimum
                # and maximum values of data seen so far by the current node
                data_sampled: ndarray = self.random_generator.uniform(
                    node.min_values,
                    node.max_values,
                    size=(node.data_size, data.shape[1]),
                )
                return self._recursive_build(
                    data_sampled, depth=node.depth, node_index=node_index
                )
            else:
                return node_index, node
        # If the current node is not a leaf, recursively update all its children
        else:
            # Partition data
            partition_indices: list[ndarray[int]] = self._split_data(
                data, node.projection_vector, node.split_values
            )
            # Recursively update children
            for i, indices in enumerate(partition_indices):
                node_index, node.children[i] = self._recursive_learn(
                    node.children[i], data[indices], node_index
                )
            return node_index, node

    def _recursive_build(
        self, data: ndarray, depth: int = 0, node_index: int = 0
    ) -> Tuple[int, OnlineIsolationNode]:
        # If there aren't enough samples to be split according to the max leaf samples or the depth limit has been
        # reached, build a leaf node
        if (
            data.shape[0]
            < self.max_leaf_samples
            * OnlineIsolationTree._get_multiplier(self.growth_criterion, depth)
            or depth >= self.depth_limit
        ):
            return node_index + 1, OnlineIsolationNode(
                data_size=data.shape[0],
                children=None,
                depth=depth,
                node_index=node_index,
                min_values=data.min(axis=0, initial=inf),
                max_values=data.max(axis=0, initial=-inf),
                projection_vector=None,
                split_values=None,
            )
        else:
            # Sample projection vector
            if self.split == "axisparallel":
                projection_vector: ndarray[float] = zeros(data.shape[1])
                projection_vector[
                    self.random_generator.choice(projection_vector.shape[0])
                ]: float = 1.0
            else:
                raise ValueError("Bad split {}".format(self.split))
            projection_vector: ndarray[float] = projection_vector / norm(
                projection_vector
            )
            # Project sampled data using projection vector
            projected_data: ndarray = data @ projection_vector
            # Sample split values
            split_values: ndarray[float] = sort(
                self.random_generator.uniform(
                    min(projected_data),
                    max(projected_data),
                    size=self.branching_factor - 1,
                )
            )
            # Partition sampled data
            partition_indices: list[ndarray[int]] = self._split_data(
                data, projection_vector, split_values
            )
            # Generate recursively children nodes
            children: ndarray[OnlineIsolationNode] = empty(
                shape=(self.branching_factor,), dtype=OnlineIsolationNode
            )
            for i, indices in enumerate(partition_indices):
                node_index, children[i] = self._recursive_build(
                    data[indices], depth + 1, node_index
                )
            return node_index + 1, OnlineIsolationNode(
                data_size=data.shape[0],
                children=children,
                depth=depth,
                node_index=node_index,
                min_values=data.min(axis=0),
                max_values=data.max(axis=0),
                projection_vector=projection_vector,
                split_values=split_values,
            )

    def _unlearn(self, data: ndarray) -> OnlineIsolationTree:
        # Subsample data in order to improve diversity among trees
        data: ndarray = data[
            self.random_generator.random(data.shape[0]) < self.subsample
        ]
        if data.shape[0] >= 1:
            # Update the counter of data seen so far
            self.data_size -= data.shape[0]
            # Adjust depth limit according to data seen so far and branching factor
            self.depth_limit: float = OnlineIsolationTree._get_random_path_length(
                self.branching_factor, self.max_leaf_samples, self.data_size
            )
            # Recursively update the tree
            self.root: OnlineIsolationNode = self._recursive_unlearn(self.root, data)
        return self

    def _recursive_unlearn(
        self, node: OnlineIsolationNode, data: ndarray
    ) -> OnlineIsolationNode:
        # Update the number of data seen so far by the current node
        node.data_size -= data.shape[0]
        # If the current node is a leaf, return it
        if node.children is None:
            return node
        # If the current node is not a leaf, try to unsplit it
        else:
            # If there are not enough samples according to max leaf samples, unsplit the node
            if (
                node.data_size
                < self.max_leaf_samples
                * OnlineIsolationTree._get_multiplier(self.growth_criterion, node.depth)
            ):
                return self._recursive_unbuild(node)
            # If there are enough samples according to max leaf samples, recursively update all its children
            else:
                # Partition data
                partition_indices: list[ndarray[int]] = self._split_data(
                    data, node.projection_vector, node.split_values
                )
                # Recursively update children
                for i, indices in enumerate(partition_indices):
                    node.children[i]: OnlineIsolationNode = self._recursive_unlearn(
                        node.children[i], data[indices]
                    )
                # Update the vectors of minimum and maximum values seen so far by the current node
                node.min_values: ndarray = vstack(
                    [node.children[i].min_values for i, _ in enumerate(node.children)]
                ).min(axis=0)
                node.max_values: ndarray = vstack(
                    [node.children[i].max_values for i, _ in enumerate(node.children)]
                ).max(axis=0)
                return node

    def _recursive_unbuild(self, node: OnlineIsolationNode) -> OnlineIsolationNode:
        # If the current node is a leaf, return it
        if node.children is None:
            return node
        # If the current node is not a leaf, unbuild it
        else:
            # Recursively unbuild children
            for i, _ in enumerate(node.children):
                node.children[i]: OnlineIsolationNode = self._recursive_unbuild(
                    node.children[i]
                )
            # Update the vectors of minimum and maximum values seen so far by the current node
            node.min_values: ndarray = vstack(
                [node.children[i].min_values for i, _ in enumerate(node.children)]
            ).min(axis=0)
            node.max_values: ndarray = vstack(
                [node.children[i].max_values for i, _ in enumerate(node.children)]
            ).max(axis=0)
            # Delete children nodes, projection vector and split values
            node.children: Optional[ndarray[OnlineIsolationNode]] = None
            node.projection_vector: Optional[ndarray[float]] = None
            node.split_values: Optional[ndarray[float]] = None
            return node

    def _predict(self, data: ndarray) -> ndarray[float]:
        # Compute depth of each sample
        if self.root:
            return self._recursive_depth_search(
                self.root, data, empty(shape=(data.shape[0],), dtype=float)
            )
        else:
            return zeros(shape=(data.shape[0],), dtype=float)

    def _recursive_depth_search(
        self, node: OnlineIsolationNode, data: ndarray, depths: ndarray[float]
    ) -> ndarray[float]:
        # If the current node is a leaf, fill the depths vector with the current depth plus a normalization factor
        if node.children is None or data.shape[0] == 0:
            depths[:] = node.depth + OnlineIsolationTree._get_random_path_length(
                self.branching_factor, self.max_leaf_samples, node.data_size
            )
        else:
            # Partition data
            partition_indices: list[ndarray[int]] = self._split_data(
                data, node.projection_vector, node.split_values
            )
            # Fill the vector of depths
            for i, indices in enumerate(partition_indices):
                depths[indices]: ndarray[float] = self._recursive_depth_search(
                    node.children[i], data[indices], depths[indices]
                )
        return depths


@dataclass
class OnlineIsolationNode:
    def __init__(
        self,
        data_size: int,
        children: Optional[ndarray[OnlineIsolationNode]],
        depth: int,
        node_index: int,
        min_values: ndarray,
        max_values: ndarray,
        projection_vector: Optional[ndarray[float]],
        split_values: Optional[ndarray[float]],
    ):
        self.data_size: int = data_size
        self.children: Optional[ndarray[OnlineIsolationNode]] = children
        self.depth: int = depth
        self.node_index: int = node_index
        self.min_values: ndarray = min_values
        self.max_values: ndarray = max_values
        self.projection_vector: Optional[ndarray[float]] = projection_vector
        self.split_values: Optional[ndarray[float]] = split_values
