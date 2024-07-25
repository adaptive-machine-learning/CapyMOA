# Library imports

from capymoa.base import MOARegressor, _extract_moa_learner_CLI
from ._arffimtdd import ARFFIMTDD


from moa.classifiers.meta import (
    AdaptiveRandomForestRegressor as MOA_AdaptiveRandomForestRegressor,
)


class AdaptiveRandomForestRegressor(MOARegressor):
    """Adaptive Random Forest Regressor

        This class implements the Adaptive Random Forest (ARF) algorithm, which is
        an ensemble regressor capable of adapting to concept drift.

        ARF is implemented in MOA (Massive Online Analysis) and provides several
        parameters for customization.

        See also :py:class:`capymoa.classifier.AdaptiveRandomForestClassifier`
        See :py:class:`capymoa.base.MOARegressor` for train and predict.

        Reference:

        `Adaptive random forests for data stream regression.
        Heitor Murilo Gomes, J. P. Barddal, L. E. B. Ferreira, A. Bifet.
        ESANN, pp. 267-272, 2018.
        <https://www.esann.org/sites/default/files/proceedings/legacy/es2018-183.pdf>`_

        Example usage:

        >>> from capymoa.datasets import Fried
        >>> from capymoa.regressor import AdaptiveRandomForestRegressor
        >>> from capymoa.evaluation import prequential_evaluation
        >>> stream = Fried()
        >>> schema = stream.get_schema()
        >>> learner = AdaptiveRandomForestRegressor(schema)
        >>> results = prequential_evaluation(stream, learner, max_instances=1000)
        >>> results["cumulative"].rmse()
        3.659072011685404
        """

    def __init__(
        self,
        schema=None,
        CLI=None,
        random_seed=1,
        tree_learner=None,
        ensemble_size=100,
        max_features=0.6,
        lambda_param=6.0,
        drift_detection_method=None,
        warning_detection_method=None,
        disable_drift_detection=False,
        disable_background_learner=False,
    ):
        """Construct an Adaptive Random Forest Regressor

        :param schema: The schema of the stream. If not provided, it will be inferred from the data.
        :param CLI: Command Line Interface (CLI) options for configuring the ARF algorithm.
            If not provided, default options will be used.
        :param random_seed: Seed for the random number generator.
        :param tree_learner: The tree learner to use. If not provided, a default Hoeffding Tree is used.
        :param ensemble_size: The number of trees in the ensemble.
        :param max_features: The maximum number of features to consider when splitting a node.
            If provided as a float between 0.0 and 1.0, it represents the percentage of features to consider.
            If provided as an integer, it specifies the exact number of features to consider.
            If provided as the string "sqrt", it indicates that the square root of the total number of features.
            If not provided, the default value is 60%.
        :param lambda_param: The lambda parameter that controls the Poisson distribution for
            the online bagging simulation.
        :param drift_detection_method: The method used for drift detection.
        :param warning_detection_method: The method used for warning detection.
        :param disable_drift_detection: Whether to disable drift detection.
        :param disable_background_learner: Whether to disable background learning.
        """

        self.moa_learner = MOA_AdaptiveRandomForestRegressor()

        # Initialize instance attributes with default values, CLI was not set.
        if CLI is None:
            if tree_learner is None:
                self.tree_learner = ARFFIMTDD(schema, grace_period=50, split_confidence=0.01)
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
                # Raise an exception with information about valid options for max_features
                raise ValueError("Invalid value for max_features. Valid options: float between 0.0 and 1.0 "
                                 "representing percentage, integer specifying exact number, or 'sqrt' for "
                                 "square root of total features.")

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

            self.moa_learner.getOptions().setViaCLIString(
                f"-l {self.tree_learner} -s {self.ensemble_size} -o {self.m_features_mode} -m \
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