from __future__ import annotations

import inspect

from capymoa.learner import MOAClassifier
import moa.classifiers.trees as moa_trees
from capymoa.stream import Schema


class EFDT(MOAClassifier):
    """Extremely Fast Decision Tree (EFDT) classifier.

    Also referred to as the Hoeffding AnyTime Tree (HATT) classifier. In practice,
    despite the name, EFDTs are typically slower than a vanilla Hoeffding Tree
    to process data. The speed differences come from the mechanism of split
    re-evaluation present in EFDT. Nonetheless, EFDT has theoretical properties
    that ensure it converges faster than the vanilla Hoeffding Tree to the structure
    that would be created by a batch decision tree model (such as Classification and
    Regression Trees - CART). Keep in mind that such propositions hold when processing
    a stationary data stream. When dealing with non-stationary data, EFDT is somewhat
    robust to concept drifts as it continually revisits and updates its internal
    decision tree structure. Still, in such cases, the Hoeffind Adaptive Tree might
    be a better option, as it was specifically designed to handle non-stationarity.


    Parameters
    ----------
    grace_period
        Number of instances a leaf should observe between split attempts.
    min_samples_reevaluate
        Number of instances a node should observe before reevaluating the best split.
    split_criterion
        Split criterion to use.</br>
        - 'gini' - Gini</br>
        - 'info_gain' - Information Gain</br>
        - 'hellinger' - Helinger Distance</br>
    confidence
        Significance level to calculate the Hoeffding bound. The significance level is given by
        `1 - delta`. Values closer to zero imply longer split decision delays.
    tau
        Threshold below which a split will be forced to break ties.
    leaf_prediction
        Prediction mechanism used at leafs.</br>
        - 'mc' - Majority Class</br>
        - 'nb' - Naive Bayes</br>
        - 'nba' - Naive Bayes Adaptive</br>
    nb_threshold
        Number of instances a leaf should observe before allowing Naive Bayes.
    numeric_attribute_observer
        The Splitter or Attribute Observer (AO) used to monitor the class statistics of numeric
        features and perform splits. Splitters are available in the `tree.splitter` module.
        Different splitters are available for classification and regression tasks. Classification
        and regression splitters can be distinguished by their property `is_target_class`.
        This is an advanced option. Special care must be taken when choosing different splitters.
        By default, `tree.splitter.GaussianSplitter` is used if `splitter` is `None`.
    binary_split
        If True, only allow binary splits.
    min_branch_fraction
        The minimum percentage of observed data required for branches resulting from split
        candidates. To validate a split candidate, at least two resulting branches must have
        a percentage of samples greater than `min_branch_fraction`. This criterion prevents
        unnecessary splits when the majority of instances are concentrated in a single branch.
    max_share_to_split
        Only perform a split in a leaf if the proportion of elements in the majority class is
        smaller than this parameter value. This parameter avoids performing splits when most
        of the data belongs to a single class.
    max_byte_size
        The max size of the tree, in bytes.
    memory_estimate_period
        Interval (number of processed instances) between memory consumption checks.
    stop_mem_management
        If True, stop growing as soon as memory limit is hit.
    remove_poor_attrs
        If True, disable poor attributes to reduce memory usage.
    disable_prepruning
        If True, disable merit-based tree pre-pruning.
    """

    MAJORITY_CLASS = 0
    NAIVE_BAYES = 1
    NAIVE_BAYES_ADAPTIVE = 2


    def __init__(
            self,
            schema: Schema | None = None,
            random_seed: int = 0,
            grace_period: int = 200,
            min_samples_reevaluate: int = 200,
            split_criterion: str = "InfoGainSplitCriterion",
            confidence: float = 1e-3,
            tie_threshold: float = 0.05,
            leaf_prediction: int = MAJORITY_CLASS,
            nb_threshold: int = 0,
            numeric_attribute_observer: str = "GaussianNumericAttributeClassObserver",
            binary_split: bool = False,
            min_branch_fraction: float = 0.01,
            max_share_to_split: float = 0.99,
            max_byte_size: float = 33554433,
            memory_estimate_period: int = 1000000,
            stop_mem_management: bool = True,
            remove_poor_attrs: bool = False,
            disable_prepruning: bool = True,
    ):
        mappings = {
            "grace_period": "-g",
            "min_samples_reevaluate": "-R",
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
            "nb_threshold": "-q"
        }

        config_str = ""
        parameters = inspect.signature(self.__init__).parameters
        for key in mappings:
            if key not in parameters:
                continue
            this_parameter = parameters[key]
            default_value = this_parameter.default
            set_value = locals()[key]
            is_bool = type(set_value) == bool
            if is_bool:
                if set_value:
                    str_extension = mappings[key] + " "
                else:
                    str_extension = ""
            else:
                str_extension = f"{mappings[key]} {set_value} "
            config_str += str_extension

        super(EFDT, self).__init__(moa_learner=moa_trees.EFDT,
                                   schema=schema,
                                   CLI=config_str,
                                   random_seed=random_seed)
