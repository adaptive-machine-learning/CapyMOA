# Library imports

from ._soknl_base_tree import SOKNLBT
from moa.classifiers.meta import SelfOptimisingKNearestLeaves as _MOA_SOKNL

from capymoa.base import MOARegressor, _extract_moa_learner_CLI


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
        >>> from capymoa.regressor import SOKNL
        >>> from capymoa.evaluation import prequential_evaluation
    >>> stream = Fried()
    >>> schema = stream.get_schema()
    >>> learner = SOKNL(schema)
    >>> results = prequential_evaluation(stream, learner, max_instances=1000)
    >>> results["cumulative"].rmse()
    3.3738337530234306
    """

    def __init__(
        self,
        schema=None,
        CLI=None,
        random_seed=1,
        tree_learner=None,
        ensemble_size=100,
        max_features=0.6,
        lambda_param=6.0,  # m_features_mode=None, m_features_per_tree_size=60,
        drift_detection_method=None,
        warning_detection_method=None,
        disable_drift_detection=False,
        disable_background_learner=False,
        disable_self_optimising=False,
        k_value=10,
    ):
        # Important: must create the MOA object before invoking the super class __init__
        self.moa_learner = _MOA_SOKNL()

        # Initialize instance attributes with default values, CLI was not set.
        if CLI is None:
            if tree_learner is None:
                self.tree_learner = SOKNLBT(
                    schema, grace_period=50, split_confidence=0.01
                )
            elif type(tree_learner) is str:
                self.tree_learner = tree_learner
            else:
                self.tree_learner = _extract_moa_learner_CLI(tree_learner)

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

            # self.m_features_mode = "(Percentage (M * (m / 100)))" if m_features_mode is None else m_features_mode
            # self.m_features_per_tree_size = m_features_per_tree_size
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
            CLI=CLI,
            random_seed=random_seed,
            moa_learner=self.moa_learner,
        )
