# Library imports

from capymoa.base import MOARegressor
from capymoa._cli import cli_str
from ._arffimtdd import ARFFIMTDD
from capymoa._cli import (
    cli_str_drift_detector,
)
from capymoa.drift.base_detector import MOADriftDetector
from moa.classifiers.meta import (
    AdaptiveRandomForestRegressor as _AdaptiveRandomForestRegressor,
)
from moa.classifiers.trees import ARFFIMTDD as J_ARFFIMTDD


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

    >>> from capymoa.datasets import FriedTiny
    >>> from capymoa.regressor import AdaptiveRandomForestRegressor
    >>> from capymoa.evaluation import prequential_evaluation
    >>> stream = FriedTiny()
    >>> schema = stream.get_schema()
    >>> learner = AdaptiveRandomForestRegressor(schema)
    >>> results = prequential_evaluation(stream, learner, max_instances=1000)
    >>> results["cumulative"].rmse()
    4.146151270393789
    """

    def __init__(
        self,
        schema=None,
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
        if isinstance(max_features, float) and 0.0 <= max_features <= 1.0:
            m_features_mode = "Percentage (M * (m / 100))"
            m_features_per_tree_size = int(max_features * 100)
        elif isinstance(max_features, int):
            m_features_mode = "Specified m (integer value)"
            m_features_per_tree_size = max_features
        elif max_features in ["sqrt"]:
            m_features_mode = "sqrt(M)+1"
            m_features_per_tree_size = -1  # or leave it unchanged
        elif max_features is None:
            m_features_mode = "Percentage (M * (m / 100))"
            m_features_per_tree_size = 60
        else:
            # Raise an exception with information about valid options for max_features
            raise ValueError(
                "Invalid value for max_features. Valid options: float between 0.0 and 1.0 "
                "representing percentage, integer specifying exact number, or 'sqrt' for "
                "square root of total features."
            )

        if tree_learner is None:
            tree_learner = ARFFIMTDD(schema, grace_period=50, split_confidence=0.01)
        if isinstance(tree_learner, ARFFIMTDD):
            tree_learner = cli_str(tree_learner.moa_learner, J_ARFFIMTDD)
        if isinstance(drift_detection_method, MOADriftDetector):
            drift_detection_method = cli_str_drift_detector(drift_detection_method)
        if isinstance(warning_detection_method, MOADriftDetector):
            warning_detection_method = cli_str_drift_detector(warning_detection_method)

        cli = [
            f"-l {tree_learner}",
            f"-s {ensemble_size}",
            f"-o '{m_features_mode}'",
            f"-m {m_features_per_tree_size}",
            f"-a {lambda_param}",
        ]

        # Optional options
        if drift_detection_method is not None:
            cli.append(f"-x {drift_detection_method}")
        if warning_detection_method is not None:
            cli.append(f"-p {warning_detection_method}")
        if disable_drift_detection:
            cli.append("-u")
        if disable_background_learner:
            cli.append("-q")

        moa_learner = _AdaptiveRandomForestRegressor()
        super().__init__(
            moa_learner=moa_learner,
            schema=schema,
            CLI=" ".join(cli),
            random_seed=random_seed,
        )
