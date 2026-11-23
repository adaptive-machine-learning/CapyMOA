# Library imports

from typing import Literal

from moa.classifiers.meta import SelfOptimisingKNearestLeaves as _MOA_SOKNL
from moa.classifiers.trees import SelfOptimisingBaseTree as _MOA_SOKNLBT

from capymoa.base import MOARegressor
from capymoa.core.moa._cli import cli_str
from capymoa.stream import Schema

from ._soknl_base_tree import SOKNLBT


class SOKNL(MOARegressor):
    """Self-Optimising K-Nearest Leaves (SOKNL) Implementation.

    SOKNL extends the AdaptiveRandomForestRegressor by limiting the number of base trees involved in predicting
    a given instance. This approach overrides the aggregation strategy used for voting, leading to more accurate
    prediction in general.

    Specifically, each leaf in the forest stores the sum of each feature and builds the "centroid" upon request.
    The centroids then are used to calculate the Euclidean distance between the incoming instance and the leaf.
    The incoming instance gets the aggregation from k trees with closer leaves as the final prediction.
    The performances of all possible k value are accessed over time and next prediction takes the best k so far.

    See also :py:class:`capymoa.regressor.AdaptiveRandomForestRegressor`
    See :py:class:`capymoa.base.MOARegressor` for train and predict.

    Reference:

    `Sun, Yibin, Bernhard Pfahringer, Heitor Murilo Gomes, and Albert Bifet.
    "SOKNL: A novel way of integrating K-nearest neighbours with adaptive random forest regression for data streams."
    Data Mining and Knowledge Discovery 36, no. 5 (2022): 2006-2032.
    <https://researchcommons.waikato.ac.nz/server/api/core/bitstreams/f91959c0-1515-44c3-bd5f-737135ee3e48/content>`_

    Example usage:

    >>> from capymoa.datasets import Fried
    >>> from capymoa.regressor import SOKNL, evaluate_regressor
    >>> stream = Fried()
    >>> schema = stream.get_schema()
    >>> learner = SOKNL(schema, random_seed=1)
    >>> results = evaluate_regressor(stream, learner, max_instances=1000)
    >>> results["cumulative"]["rmse"]
    3.4334357446039427
    """

    def __init__(
        self,
        schema: Schema,
        tree_learner: MOARegressor | str | None = None,
        ensemble_size: int = 100,
        max_features: float | Literal["sqrt"] | None = 0.6,
        lambda_param: float = 6.0,
        drift_detection_method: str | None = None,
        warning_detection_method: str | None = None,
        disable_drift_detection: bool = False,
        disable_background_learner: bool = False,
        disable_self_optimising: bool = False,
        k_value: int = 10,
        random_seed: int | None = None,
    ) -> None:
        """Construct a SOKNL regressor.

        :param schema: Describes the stream's structure and types.
        :param tree_learner: Base tree or MOA CLI string. Defaults to a
            :class:`~capymoa.regressor.SOKNLBT` with grace period 50 and split
            confidence 0.01.
        :param ensemble_size: The number of trees in the ensemble.
        :param max_features: The subspace size for each ensemble member.
            If a float between 0.0 and 1.0, it is the fraction of features (0.6 is 60%).
            If an integer, it is the exact number of features.
            If ``"sqrt"``, it is the square root of the number of features plus 1.
            If ``None``, 60% of the features are used.
        :param lambda_param: The lambda parameter that controls the Poisson
            distribution for the online bagging simulation.
        :param drift_detection_method: The method used for drift detection (CLI
            string). If not provided, ADWIN is used.
        :param warning_detection_method: The method used for warning detection (CLI
            string). If not provided, ADWIN is used.
        :param disable_drift_detection: Whether to disable drift detection (and
            background learners).
        :param disable_background_learner: Whether to disable background learning.
        :param disable_self_optimising: Whether to use the fixed ``k_value`` instead
            of self-optimising.
        :param k_value: The number of nearest leaves to use when self-optimising is
            disabled.
        :param random_seed: Seed for all random choices. ``None`` is a fresh random seed each run.
        """
        # Important: must create the MOA object before invoking the super class __init__
        self.moa_learner = _MOA_SOKNL()

        if tree_learner is None:
            tree_learner = SOKNLBT(schema, grace_period=50, split_confidence=0.01)
        # MOA resolves -l class names relative to SelfOptimisingBaseTree.
        self.tree_learner = (
            tree_learner
            if isinstance(tree_learner, str)
            else cli_str(tree_learner.moa_learner, _MOA_SOKNLBT)
        )

        self.ensemble_size = ensemble_size

        self.max_features = max_features
        if isinstance(self.max_features, float) and 0.0 <= self.max_features <= 1.0:
            self.m_features_mode = "(Percentage (M * (m / 100)))"
            self.m_features_per_tree_size = int(self.max_features * 100)
        elif isinstance(self.max_features, int):
            self.m_features_mode = "(Specified m (integer value))"
            self.m_features_per_tree_size = max_features
        elif self.max_features in ["sqrt"]:
            self.m_features_mode = "(sqrt(M)+1)"
            self.m_features_per_tree_size = -1  # or leave it unchanged
        elif self.max_features is None:
            self.m_features_mode = "(Percentage (M * (m / 100)))"
            self.m_features_per_tree_size = 60
        else:
            # Handle other cases or raise an exception if needed
            raise ValueError("Invalid value for max_features")

        self.lambda_param = lambda_param
        self.drift_detection_method = (
            "(ADWINChangeDetector -a 1.0E-3)"
            if drift_detection_method is None
            else drift_detection_method
        )
        self.warning_detection_method = (
            "(ADWINChangeDetector -a 1.0E-2)"
            if warning_detection_method is None
            else warning_detection_method
        )
        self.disable_drift_detection = disable_drift_detection
        self.disable_background_learner = disable_background_learner

        self.disable_self_optimising = disable_self_optimising
        self.k_value = k_value

        self.moa_learner.getOptions().setViaCLIString(
            f"-l {self.tree_learner} -s {self.ensemble_size} {'-f' if self.disable_self_optimising else ''} -k {self.k_value} -o {self.m_features_mode} -m \
            {self.m_features_per_tree_size} -a {self.lambda_param} -x {self.drift_detection_method} -p \
            {self.warning_detection_method} {'-u' if self.disable_drift_detection else ''}  {'-q' if self.disable_background_learner else ''}"
        )
        self.moa_learner.prepareForUse()
        self.moa_learner.resetLearning()

        super().__init__(
            schema=schema,
            random_seed=random_seed,
            moa_learner=self.moa_learner,
        )
