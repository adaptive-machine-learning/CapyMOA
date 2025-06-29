from __future__ import annotations
from typing import Union

from capymoa.base import MOAClassifier
from capymoa.splitcriteria import SplitCriterion, _split_criterion_to_cli_str
from capymoa.stream import Schema
from capymoa._utils import build_cli_str_from_mapping_and_locals, _leaf_prediction

import moa.classifiers.trees as moa_trees


class EFDT(MOAClassifier):
    """Extremely Fast Decision Tree.

    Extremely Fast Decision Tree (EFDT) [#0]_ is a decision tree classifier. Also
    referred to as the Hoeffding AnyTime Tree (HATT) classifier. In practice,
    despite the name, EFDTs are typically slower than a vanilla Hoeffding Tree
    to process data. The speed differences come from the mechanism of split re-
    evaluation present in EFDT. Nonetheless, EFDT has theoretical properties
    that ensure it converges faster than the vanilla Hoeffding Tree to the
    structure that would be created by a batch decision tree model (such as
    Classification and Regression Trees - CART). Keep in mind that such
    propositions hold when processing a stationary data stream. When dealing
    with non-stationary data, EFDT is somewhat robust to concept drifts as it
    continually revisits and updates its internal decision tree structure.
    Still, in such cases, the Hoeffding Adaptive Tree might be a better option,
    as it was specifically designed to handle non-stationarity.

    >>> from capymoa.classifier import EFDT
    >>> from capymoa.datasets import ElectricityTiny
    >>> from capymoa.evaluation import prequential_evaluation
    >>>
    >>> stream = ElectricityTiny()
    >>> classifier = EFDT(stream.get_schema())
    >>> results = prequential_evaluation(stream, classifier, max_instances=1000)
    >>> print(f"{results['cumulative'].accuracy():.1f}")
    84.4

    .. [#0] `Extremely fast decision tree. Manapragada, Chaitanya, G. I. Webb, M.
             Salehi. ACM SIGKDD, pp. 1953-1962, 2018.
             <https://dl.acm.org/doi/abs/10.1145/3219819.3220005>`_
    """

    def __init__(
        self,
        schema: Schema | None = None,
        random_seed: int = 0,
        grace_period: int = 200,
        min_samples_reevaluate: int = 200,
        split_criterion: Union[str, SplitCriterion] = "InfoGainSplitCriterion",
        confidence: float = 1e-3,
        tie_threshold: float = 0.05,
        leaf_prediction: str = "NaiveBayesAdaptive",
        nb_threshold: int = 0,
        numeric_attribute_observer: str = "GaussianNumericAttributeClassObserver",
        binary_split: bool = False,
        max_byte_size: float = 33554433,
        memory_estimate_period: int = 1000000,
        stop_mem_management: bool = True,
        remove_poor_attrs: bool = False,
        disable_prepruning: bool = True,
    ):
        """Construct an Extremely Fast Decision Tree (EFDT) Classifier

        :param schema: The schema of the stream.
        :param random_seed: The random seed passed to the MOA learner.
        :param grace_period: Number of instances a leaf should observe between split attempts.
        :param min_samples_reevaluate: Number of instances a node should observe before re-evaluating the best split.
        :param split_criterion: Split criterion to use. Defaults to `InfoGainSplitCriterion`.
        :param confidence: Significance level to calculate the Hoeffding bound. The significance level is given by
            `1 - delta`. Values closer to zero imply longer split decision delays.
        :param tie_threshold: Threshold below which a split will be forced to break ties.
        :param leaf_prediction: Prediction mechanism used at the leaves
            ("MajorityClass" or 0, "NaiveBayes" or 1, "NaiveBayesAdaptive" or 2).
        :param nb_threshold: Number of instances a leaf should observe before allowing Naive Bayes.
        :param numeric_attribute_observer: The Splitter or Attribute Observer (AO) used to monitor the class statistics
            of numeric features and perform splits.
        :param binary_split: If True, only allow binary splits.
        :param max_byte_size: The max size of the tree, in bytes.
        :param memory_estimate_period: Interval (number of processed instances) between memory consumption checks.
        :param stop_mem_management: If True, stop growing as soon as memory limit is hit.
        :param remove_poor_attrs: If True, disable poor attributes to reduce memory usage.
        :param disable_prepruning: If True, disable merit-based tree pre-pruning.
        """

        mapping = {
            "grace_period": "-g",
            "min_samples_reevaluate": "-R",
            "split_criterion": "-s",
            "confidence": "-c",
            "tie_threshold": "-t",
            "leaf_prediction": "-l",
            "nb_threshold": "-q",
            "numeric_attribute_observer": "-n",
            "binary_split": "-b",
            "max_byte_size": "-m",
            "memory_estimate_period": "-e",
            "stop_mem_management": "-z",
            "remove_poor_attrs": "-r",
            "disable_prepruning": "-p",
        }
        split_criterion = _split_criterion_to_cli_str(split_criterion)
        leaf_prediction = _leaf_prediction(leaf_prediction)
        config_str = build_cli_str_from_mapping_and_locals(mapping, locals())
        super(EFDT, self).__init__(
            moa_learner=moa_trees.EFDT,
            schema=schema,
            CLI=config_str,
            random_seed=random_seed,
        )
