# SPDX-License-Identifier: BSD-3-Clause
from typing import Literal, Union

from openmoa._prepare_jpype import _start_jpype
from openmoa.base import MOAClassifier
from openmoa.stream import Schema

_start_jpype()
from moa.classifiers.trees import PLASTIC as _PLASTIC
from openmoa.splitcriteria import SplitCriterion, _split_criterion_to_cli_str

class PLASTIC(MOAClassifier):
    """PLASTIC classifier.

    PLASTIC [#f1]_ is an incremental decision tree that restructures the otherwise
    pruned subtree. PLASTIC improves upon Extremely Fast Decision Trees (EFDT) by
    not only revisiting previously splits but also trying to maintain as much as
    possible of the structure once a split is redone. This process is possible
    because of the decision tree plasticity: one can alter a tree’s structure without
    affecting its predictions.

    >>> from openmoa.classifier import PLASTIC
    >>> PLASTIC.__name__
    'PLASTIC'

    .. [#f1] Heyden, Marco, et al. "Leveraging plasticity in incremental decision trees."
             Joint European Conference on Machine Learning and Knowledge Discovery in
             Databases. Cham: Springer Nature Switzerland, 2024.
    """

    def __init__(
        self,
        schema: Schema,
        grace_period: int = 200,
        reevaluation_period: int = 200,
        nominal_estimator: str = "NominalAttributeClassObserver",
        split_criterion: Union[str, SplitCriterion] = "InfoGainSplitCriterion",
        split_confidence: float = 1e-07,
        tie_threshold: float = 0.05,
        tie_threshold_reevaluation: float = 0.05,
        rel_min_delta_g: float = 0.5,
        binary_splits: bool = False,
        leaf_prediction: Literal["MC", "NB", "NBA"] = "NBA",
        max_depth: int = 20,
        max_branch_length: int = 5,
    ) -> None:
        """Construct PLASTIC classifier.

        :param grace_period: The number of instances a leaf should observe between split
            attempts.
        :param reevaluation_period: The number of instances an internal node should
            observe between re-evaluation attempts.
        :param nominal_estimator: Nominal estimator to use.
        :param split_criterion: Split criterion to use.
        :param split_confidence: The allowable error in split decision when using fixed
            confidence. Values closer to 0 will take longer to decide.
        :param tie_threshold: Threshold below which a split will be forced to break
            ties.
        :param tie_threshold_reevaluation: Threshold below which a split will be forced
            to break ties during reevaluation.
        :param rel_min_delta_g: Relative minimum information gain to split a tie during
            reevaluation.
        :param binary_splits: Only allow binary splits.
        :param leaf_prediction: Leaf prediction to use.

            * ``MC``: Majority class
            * ``NB``: Naive Bayes
            * ``NBA``: Naive Bayes Adaptive
        :param max_depth: Maximum allowed depth of tree.
        :param max_branch_length: Maximum allowed length of branches during
            restructuring.
        """

        cli = []
        cli += [f"-g {grace_period}"]
        cli += [f"-R {reevaluation_period}"]
        cli += [f"-d {nominal_estimator}"]
        cli += [f"-s {_split_criterion_to_cli_str(split_criterion)}"]
        cli += [f"-c {split_confidence}"]
        cli += [f"-t {tie_threshold}"]
        cli += [f"-T {tie_threshold_reevaluation}"]
        cli += [f"-G {rel_min_delta_g}"]
        cli += ["-b"] if binary_splits else []
        cli += [f"-l {leaf_prediction}"]
        cli += [f"-D {max_depth}"]
        cli += [f"-B {max_branch_length}"]

        super().__init__(moa_learner=_PLASTIC, schema=schema, CLI=" ".join(cli))
