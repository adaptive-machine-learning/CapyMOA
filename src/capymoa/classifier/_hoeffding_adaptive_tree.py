from __future__ import annotations
from typing import Union

# from capymoa.base import MOAClassifier
from capymoa.classifier import HoeffdingTree
from capymoa.splitcriteria import SplitCriterion, _split_criterion_to_cli_str
from capymoa.stream import Schema
from capymoa._utils import build_cli_str_from_mapping_and_locals, _leaf_prediction

import moa.classifiers.trees as moa_trees


class HoeffdingAdaptiveTree(HoeffdingTree):
    """Hoeffding Adaptive Tree (HAT).

    HAT [#bifet2009]_ uses :class:`~capymoa.drift.detectors.ADWIN` to track the
    performance of its tree branches allowing it to adapt to concept drift.

    >>> from capymoa.datasets import ElectricityTiny
    >>> from capymoa.classifier import HoeffdingAdaptiveTree
    >>> from capymoa.evaluation import prequential_evaluation
    >>> stream = ElectricityTiny()
    >>> schema = stream.get_schema()
    >>> learner = HoeffdingAdaptiveTree(schema)
    >>> results = prequential_evaluation(stream, learner, max_instances=1000)
    >>> results["cumulative"].accuracy()
    84.1

    ..  [#bifet2009] Bifet, A. and Gavalda, R., 2009. Adaptive learning from
        evolving data streams. In Advances in Intelligent Data Analysis VIII:
        8th International Symposium on Intelligent Data Analysis, IDA 2009,
        Lyon, France, August 31-September 2, 2009. Proceedings 8 (pp. 249-260).
        Springer Berlin Heidelberg.
        https://link.springer.com/chapter/10.1007/978-3-642-03915-7_22
    """

    def __init__(
        self,
        schema: Schema,
        random_seed: int = 0,
        grace_period: int = 200,
        split_criterion: Union[str, SplitCriterion] = "InfoGainSplitCriterion",
        confidence: float = 1e-3,
        tie_threshold: float = 0.05,
        leaf_prediction: Union[str, int] = "NaiveBayesAdaptive",
        nb_threshold: int = 0,
        numeric_attribute_observer: str = "GaussianNumericAttributeClassObserver",
        binary_split: bool = False,
        max_byte_size: float = 33554433,
        memory_estimate_period: int = 1000000,
        stop_mem_management: bool = True,
        remove_poor_attrs: bool = False,
        disable_prepruning: bool = True,
    ):
        """Hoeffding Adaptive Tree (HAT) classifier.

        :param schema: the schema of the stream.
        :param random_seed: the random seed passed to the moa learner.
        :param grace_period: the number of instances a leaf should observe between split attempts.
        :param split_criterion: the split criterion to use. Defaults to `InfoGainSplitCriterion`.
        :param confidence: the confidence level to calculate the Hoeffding Bound (1 - delta).
            Defaults to `1e-3`. Values closer to zero imply longer split decision delays.
        :param tie_threshold: the threshold below which a split will be forced to break ties.
        :param leaf_prediction: the Prediction mechanism used at leafs.</br>
            - 0 - Majority Class</br>
            - 1 - Naive Bayes</br>
            - 2 - Naive Bayes Adaptive</br>
        :param nb_threshold: the number of instances a leaf should observe before allowing Naive Bayes.
        :param numeric_attribute_observer: the Splitter or Attribute Observer (AO) used to
            monitor the class statistics of numeric features and perform splits.
        :param binary_split: If True, only allow binary splits.
        :param max_byte_size: the max size of the tree, in bytes.
        :param memory_estimate_period: Interval (number of processed instances) between memory consumption checks.
        :param stop_mem_management: If True, stop growing as soon as memory limit is hit.
        :param remove_poor_attrs: If True, disable poor attributes to reduce memory usage.
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
            moa_learner=moa_trees.HoeffdingAdaptiveTree,
            schema=schema,
            CLI=config_str,
            random_seed=random_seed,
        )
