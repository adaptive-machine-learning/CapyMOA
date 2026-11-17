from __future__ import annotations

import moa.classifiers.trees as moa_trees

from capymoa._utils import _leaf_prediction
from capymoa.base import MOAClassifier
from capymoa.core.moa.splitcriteria import (
    SplitCriterion,
    _split_criterion_to_cli_str,
)
from capymoa.drift.base_detector import BaseDriftDetector
from capymoa.drift.detectors import ADWIN
from capymoa.stream import Schema


class LAST(MOAClassifier):
    """Local Adaptive Streaming Tree.

    Local Adaptive Streaming Tree (LAST) [#l1]_ is an incremental decision tree
    with adaptive splitting mechanisms. LAST maintains a change detector at each
    leaf and splits this node if a change is detected in the error or the leaf's
    data distribution.

    An appealing feature of LAST is that users do not need to specify
    the Grace Period, Tau threshold and confidence hyperparameters
    as in Hoeffding Trees [#l2]_.

    >>> from capymoa.classifier import LAST
    >>> from capymoa.datasets import ElectricityTiny
    >>> from capymoa.classifier import evaluate_classifier
    >>> from capymoa.drift.detectors import HDDMAverage
    >>>
    >>> stream = ElectricityTiny()
    >>> classifier = LAST(stream.get_schema(), change_detector=HDDMAverage())
    >>> results = evaluate_classifier(stream, classifier, max_instances=1000)
    >>> print(f"{results['cumulative']['accuracy']:.1f}")
    88.6

    .. [#l1] Daniel Nowak Assis, Jean Paul Barddal, and Fabrício Enembreck.
       Just Change on Change: Adaptive Splitting Time for Decision Trees in Data Stream
       Classification. 39th ACM/SIGAPP Symposium on Applied Computing (SAC '24).

    .. [#l2] Daniel Nowak Assis, Jean Paul Barddal, and Fabrício Enembreck.
       Behavioral insights of adaptive splitting decision trees in evolving data stream
       classification. Knowledge and Information Systems, 2025.
    """

    def __init__(
        self,
        schema: Schema | None = None,
        split_criterion: str | SplitCriterion = "InfoGainSplitCriterion",
        change_detector: BaseDriftDetector | None = None,
        monitor_distribution=False,
        leaf_prediction: int = "NaiveBayesAdaptive",
        nb_threshold: int = 0,
        numeric_attribute_observer: str = "GaussianNumericAttributeClassObserver",
        binary_split: bool = False,
        max_byte_size: float = 33554433,
        memory_estimate_period: int = 1000000,
        stop_mem_management: bool = True,
        remove_poor_attrs: bool = False,
        disable_prepruning: bool = True,
        random_seed: int | None = None,
    ):
        """Construct Local Adaptive Streaming Tree.

        :param schema: Describes the stream's structure and types.
        :param split_criterion: Split criterion to use.
        :param change_detector: The change detector created at leaf nodes
            that determines splitting time upon increase in error or impurity of class
            distribution.
        :param monitor_distribution: If True, change detector monitors class distribution impurity.
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
        :param random_seed: Seed for all random choices. ``None`` is a fresh random seed each run.
        """
        if change_detector is None:
            change_detector = ADWIN()

        options = {
            "maxByteSize": max_byte_size,
            "numericEstimator": numeric_attribute_observer,
            "memoryEstimatePeriod": memory_estimate_period,
            "splitCriterion": _split_criterion_to_cli_str(split_criterion),
            "changeDetectionMethodOption": change_detector,
            "distributionOption": monitor_distribution,
            "binarySplits": binary_split,
            "stopMemManagement": stop_mem_management,
            "removePoorAtts": remove_poor_attrs,
            "noPrePrune": disable_prepruning,
            "leafprediction": _leaf_prediction(leaf_prediction),
            "nbThreshold": nb_threshold,
        }
        super().__init__(
            moa_learner=moa_trees.LAST,
            schema=schema,
            random_seed=random_seed,
            options=options,
        )
