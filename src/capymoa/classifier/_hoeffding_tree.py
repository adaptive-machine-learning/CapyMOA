from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Literal, Sequence, Union

import numpy as np

from capymoa.base import MOAClassifier
from capymoa.splitcriteria import SplitCriterion, _split_criterion_to_cli_str
from capymoa.stream import Schema
from capymoa._utils import build_cli_str_from_mapping_and_locals, _leaf_prediction

from capymoa.visualization import export_hoeffding_tree_to_dot

import moa.classifiers.trees as moa_trees

MissingValuePolicy = Literal["default", "random", "all"]
TreeEdge = tuple[Any, int, Any]


@dataclass(frozen=True)
class TreePredictionTrace:
    """Prediction votes and tree path used to produce them."""

    votes: np.ndarray
    vote_node: Any | None = None
    nodes: tuple[Any, ...] = ()
    edges: tuple[TreeEdge, ...] = ()


def _validate_missing_value_policy(
    policy: str,
) -> MissingValuePolicy:
    valid_policies = ("default", "random", "all")
    if policy not in valid_policies:
        raise ValueError(
            "Invalid value for missing_value_policy, valid options are "
            "'default', 'random', or 'all'."
        )
    return policy  # type: ignore[return-value]


class HoeffdingTree(MOAClassifier):
    """Hoeffding Tree.

    Hoeffding Tree (VFDT) [#0]_ is a tree classifier classifier. A Hoeffding
    tree is an incremental, anytime decision tree induction algorithm that is
    capable of learning from massive data streams, assuming that the
    distribution generating examples does not change over time. Hoeffding trees
    exploit the fact that a small sample can often be enough to choose an
    optimal splitting attribute. This idea is supported mathematically by the
    Hoeffding bound, which quantiﬁes the number of observations (in our case,
    examples) needed to estimate some statistics within a prescribed precision
    (in our case, the goodness of an attribute).

    A theoretically appealing feature of Hoeffding Trees not shared by other
    incremental decision tree learners is that it has sound guarantees of
    performance. Using the Hoeffding bound one can show that its output is
    asymptotically nearly identical to that of a non-incremental learner using
    inﬁnitely many examples.

    >>> from capymoa.classifier import HoeffdingTree
    >>> from capymoa.datasets import ElectricityTiny
    >>> from capymoa.evaluation import prequential_evaluation
    >>>
    >>> stream = ElectricityTiny()
    >>> classifier = HoeffdingTree(stream.get_schema())
    >>> results = prequential_evaluation(stream, classifier, max_instances=1000)
    >>> print(f"{results['cumulative'].accuracy():.1f}")
    84.4

    .. [#0] `G. Hulten, L. Spencer, and P. Domingos. Mining time-changing data streams.
             In KDD’01, pages 97–106, San Francisco, CA, 2001. ACM Press.
             <https://dl.acm.org/doi/10.1145/502512.502529>`_
    """

    def __init__(
        self,
        schema: Schema | None = None,
        random_seed: int = 0,
        grace_period: int = 200,
        split_criterion: Union[str, SplitCriterion] = "InfoGainSplitCriterion",
        confidence: float = 1e-3,
        tie_threshold: float = 0.05,
        leaf_prediction: int = "NaiveBayesAdaptive",
        nb_threshold: int = 0,
        numeric_attribute_observer: str = "GaussianNumericAttributeClassObserver",
        binary_split: bool = False,
        max_byte_size: float = 33554433,
        memory_estimate_period: int = 1000000,
        stop_mem_management: bool = True,
        remove_poor_attrs: bool = False,
        disable_prepruning: bool = True,
        missing_value_policy: MissingValuePolicy = "default",
    ):
        """Construct Hoeffding Tree.

        :param schema: Stream schema.
        :param random_seed: Seed for reproducibility.
        :param grace_period: Number of instances a leaf should observe between split
            attempts.
        :param split_criterion: Split criterion to use.
        :param confidence: Significance level to calculate the Hoeffding bound. The
            significance level is given by `1 - delta`. Values closer to zero imply
            longer split decision delays.
        :param tie_threshold: Threshold below which a split will be forced to break
            ties.
        :param leaf_prediction: Prediction mechanism used at leafs.
        :param nb_threshold: Number of instances a leaf should observe before allowing
            Naive Bayes.
        :param numeric_attribute_observer: The Splitter or Attribute Observer (AO) used
            to monitor the class statistics of numeric features and perform splits.
        :param binary_split: If True, only allow binary splits.
        :param max_byte_size: The max size of the tree, in bytes.
        :param memory_estimate_period: Interval (number of processed instances) between
            memory consumption checks.
        :param stop_mem_management: If True, stop growing as soon as memory limit is
            hit.
        :param remove_poor_attrs: If True, disable poor attributes to reduce memory
            usage.
        :param disable_prepruning: If True, disable merit-based tree pre-pruning.
        :param missing_value_policy: Prediction-time policy used when a
            split attribute needed for traversal is missing. ``"default"``
            delegates to MOA's default behavior, ``"random"`` follows one
            random child, and ``"all"`` combines votes from all reachable
            children.
        """
        self.missing_value_policy = _validate_missing_value_policy(missing_value_policy)
        self._prediction_rng = np.random.default_rng(random_seed)
        mapping = {
            "grace_period": "-g",
            "max_byte_size": "-m",
            "numeric_attribute_observer": "-n",
            "memory_estimate_period": "-e",
            "split_criterion": "-s",
            "confidence": "-c",
            "tie_threshold": "-t",
            "binary_split": "-b",
            "stop_mem_management": "-z",
            "remove_poor_attrs": "-r",
            "disable_prepruning": "-p",
            "leaf_prediction": "-l",
            "nb_threshold": "-q",
        }
        split_criterion = _split_criterion_to_cli_str(split_criterion)
        leaf_prediction = _leaf_prediction(leaf_prediction)
        config_str = build_cli_str_from_mapping_and_locals(mapping, locals())
        super(HoeffdingTree, self).__init__(
            moa_learner=moa_trees.HoeffdingTree,
            schema=schema,
            CLI=config_str,
            random_seed=random_seed,
        )

    def predict_proba(self, instance):
        if self.missing_value_policy == "default" or not self._has_missing_features(
            instance
        ):
            return super().predict_proba(instance)

        root = self.get_tree_root()
        if root is None:
            return super().predict_proba(instance)

        votes = self.trace_prediction_path(instance.java_instance.getData(), root).votes
        return self._normalize_votes(votes)

    def get_tree_root(self):
        """Return the underlying real MOA tree root."""
        return self.moa_learner.getTreeRoot()

    def export_tree_to_dot(
        self,
        sample_instance=None,
        title: str = "Real MOA Hoeffding Tree",
        include_leaf_votes: bool = True,
        highlight_path: bool = False,
        require_missing_path: bool = False,
    ) -> str:
        """Export the learned real MOA tree to DOT."""
        return export_hoeffding_tree_to_dot(
            self,
            sample_instance=sample_instance,
            title=title,
            include_leaf_votes=include_leaf_votes,
            highlight_path=highlight_path,
            require_missing_path=require_missing_path,
        )

    @staticmethod
    def _has_missing_features(instance) -> bool:
        return bool(np.isnan(np.asarray(instance.x, dtype=float)).any())

    @staticmethod
    def _normalize_votes(votes: Sequence[float] | np.ndarray):
        votes = np.asarray(votes, dtype=np.float64)
        total = votes.sum()
        if votes.size == 0 or total <= 1e-2 or np.isnan(total) or np.isinf(total):
            return None
        return votes / total

    def trace_prediction_path(self, java_instance, node=None) -> TreePredictionTrace:
        """Return votes and tree path used by the missing-value prediction policy."""
        if node is None:
            node = self.get_tree_root()
        return self._trace_prediction_path(java_instance, node)

    def _trace_prediction_path(self, java_instance, node) -> TreePredictionTrace:
        if node is None:
            return TreePredictionTrace(votes=np.array([], dtype=np.float64))

        if node.isLeaf():
            return TreePredictionTrace(
                votes=self._node_votes(node, java_instance),
                vote_node=node,
                nodes=(node,),
            )

        split_test = node.getSplitTest()
        if not split_test.resultKnownForInstance(java_instance):
            if self.missing_value_policy == "default":
                return TreePredictionTrace(
                    votes=self._node_votes(node, java_instance),
                    vote_node=node,
                    nodes=(node,),
                )

            children = []
            for child_idx in range(node.numChildren()):
                child = node.getChild(child_idx)
                if child is not None:
                    children.append((child_idx, child))
            if not children:
                return TreePredictionTrace(
                    votes=self._node_votes(node, java_instance),
                    vote_node=node,
                    nodes=(node,),
                )

            if self.missing_value_policy == "random":
                for pick_pos in self._prediction_rng.permutation(len(children)):
                    branch_idx, child = children[int(pick_pos)]
                    trace = self._trace_prediction_path(java_instance, child)
                    if self._has_usable_votes(trace.votes):
                        return TreePredictionTrace(
                            votes=trace.votes,
                            vote_node=trace.vote_node,
                            nodes=(node, *trace.nodes),
                            edges=((node, branch_idx, child), *trace.edges),
                        )
                return TreePredictionTrace(
                    votes=self._node_votes(node, java_instance),
                    vote_node=node,
                    nodes=(node,),
                )

            child_traces = [
                self._trace_prediction_path(java_instance, child)
                for _, child in children
            ]
            edges = tuple(
                (node, branch_idx, child) for branch_idx, child in children
            )
            return TreePredictionTrace(
                votes=self._sum_votes([trace.votes for trace in child_traces]),
                vote_node=None,
                nodes=(node, *(n for trace in child_traces for n in trace.nodes)),
                edges=(*edges, *(e for trace in child_traces for e in trace.edges)),
            )

        branch_idx = int(split_test.branchForInstance(java_instance))
        child = node.getChild(branch_idx)
        if child is None:
            return TreePredictionTrace(
                votes=self._node_votes(node, java_instance),
                vote_node=node,
                nodes=(node,),
            )

        trace = self._trace_prediction_path(java_instance, child)
        return TreePredictionTrace(
            votes=trace.votes,
            vote_node=trace.vote_node,
            nodes=(node, *trace.nodes),
            edges=((node, branch_idx, child), *trace.edges),
        )

    def _node_votes(self, node, java_instance) -> np.ndarray:
        return np.asarray(
            node.getClassVotes(java_instance, self.moa_learner), dtype=np.float64
        )

    @staticmethod
    def _has_usable_votes(votes: Sequence[float] | np.ndarray) -> bool:
        votes = np.asarray(votes, dtype=np.float64)
        total = votes.sum()
        return bool(
            votes.size > 0
            and total > 1e-2
            and not np.isnan(total)
            and not np.isinf(total)
        )

    @staticmethod
    def _sum_votes(vote_arrays: Sequence[np.ndarray]) -> np.ndarray:
        max_len = max((len(votes) for votes in vote_arrays), default=0)
        if max_len == 0:
            return np.array([], dtype=np.float64)

        combined = np.zeros(max_len, dtype=np.float64)
        for votes in vote_arrays:
            padded_votes = np.zeros(max_len, dtype=np.float64)
            padded_votes[: len(votes)] = votes
            combined += padded_votes
        return combined
