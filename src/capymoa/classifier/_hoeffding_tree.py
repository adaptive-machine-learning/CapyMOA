from __future__ import annotations
from typing import Union

from capymoa.base import MOAClassifier
from capymoa.splitcriteria import SplitCriterion, _split_criterion_to_cli_str
from capymoa.stream import Schema
from capymoa._utils import build_cli_str_from_mapping_and_locals, _leaf_prediction

import moa.classifiers.trees as moa_trees


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
        """
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
