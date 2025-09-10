from typing import Optional
from capymoa.base import (
    MOAClassifier,
    _extract_moa_learner_CLI,
    _extract_moa_drift_detector_CLI,
)

from capymoa.base._classifier import Classifier
from capymoa.drift.base_detector import BaseDriftDetector
from capymoa.drift.detectors import (
    ADWIN,
)

from capymoa.stream._stream import Schema
from moa.classifiers.meta import AdaptiveRandomForest as _MOA_AdaptiveRandomForest
from moa.classifiers.meta.minibatch import (
    AdaptiveRandomForestMB as _MOA_AdaptiveRandomForestMB,
)
import os


class AdaptiveRandomForestClassifier(MOAClassifier):
    """Adaptive Random Forest.

    Adaptive Random Forest (ARF) [#0]_ is a decsion tree ensemble classifier.
    The 3 most important aspects of this ensemble classifier are: (1) inducing
    diversity through resampling; (2) inducing diversity through randomly
    selecting subsets of features for node splits; (3) drift detectors per base
    tree, which cause selective resets in response to drifts. It also allows
    training background trees, which start training if a warning is detected and
    replace the active tree if the warning escalates to a drift.

    >>> from capymoa.classifier import AdaptiveRandomForestClassifier
    >>> from capymoa.datasets import ElectricityTiny
    >>> from capymoa.evaluation import prequential_evaluation
    >>>
    >>> stream = ElectricityTiny()
    >>> classifier = AdaptiveRandomForestClassifier(stream.get_schema())
    >>> results = prequential_evaluation(stream, classifier, max_instances=1000)
    >>> print(f"{results['cumulative'].accuracy():.1f}")
    87.9

    .. [#0] `Gomes, H. M., Bifet, A., Read, J., Barddal, J. P., Enembreck, F.,
             Pfharinger, B., ... & Abdessalem, T. (2017). Adaptive random forests for
             evolving data stream classification. Machine Learning, 106, 1469-1495.
             <https://link.springer.com/content/pdf/10.1007/s10994-017-5642-8.pdf>`_
    """

    def __init__(
        self,
        schema: Optional[Schema] = None,
        CLI: Optional[str] = None,
        random_seed: int = 1,
        base_learner: Optional[Classifier] = None,
        ensemble_size: int = 100,
        max_features: float = 0.6,
        lambda_param: float = 6.0,
        minibatch_size: Optional[int] = None,
        number_of_jobs: int = 1,
        drift_detection_method: Optional[BaseDriftDetector] = None,
        warning_detection_method: Optional[BaseDriftDetector] = None,
        disable_weighted_vote: bool = False,
        disable_drift_detection: bool = False,
        disable_background_learner: bool = False,
    ):
        """Construct ARF.

        :param schema: The schema of the stream. If not provided, it will be inferred from the data.
        :param CLI: Command Line Interface (CLI) options for configuring the ARF algorithm.
            If not provided, default options will be used.
        :param random_seed: Seed for the random number generator.
        :param base_learner: The base learner to use. If not provided, a default Hoeffding Tree is used.
        :param ensemble_size: The number of trees in the ensemble.
        :param max_features: The maximum number of features to consider when splitting a node.
            If provided as a float between 0.0 and 1.0, it represents the percentage of features to consider.
            If provided as an integer, it specifies the exact number of features to consider.
            If provided as the string "sqrt", it indicates that the square root of the total number of features.
            If not provided, the default value is 60%.
        :param lambda_param: The lambda parameter that controls the Poisson distribution for the online bagging simulation.
        :param minibatch_size: The number of instances that a learner must accumulate before training.
        :param number_of_jobs: The number of parallel jobs to run during the execution of the algorithm.
            By default, the algorithm executes tasks sequentially (i.e., with `number_of_jobs=1`).
            Increasing the `number_of_jobs` can lead to faster execution on multi-core systems.
            However, setting it to a high value may consume more system resources and memory.
            This implementation is designed to be embarrassingly parallel, meaning that the algorithm's computations
            can be efficiently distributed across multiple processing units without sacrificing predictive
            performance. It's recommended to experiment with different values to find the optimal setting based on
            the available hardware resources and the nature of the workload.
        :param drift_detection_method: The method used for drift detection.
        :param warning_detection_method: The method used for warning detection.
        :param disable_weighted_vote: Whether to disable weighted voting.
        :param disable_drift_detection: Whether to disable drift detection.
        :param disable_background_learner: Whether to disable background learning.
        """

        if CLI is None:
            self.base_learner = (
                "(ARFHoeffdingTree -e 2000000 -g 50 -c 0.01)"
                if base_learner is None
                else _extract_moa_learner_CLI(base_learner)
            )
            self.base_learner = self.base_learner.replace("trees.", "")
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
                raise ValueError(
                    "Invalid value for max_features. Valid options: float between 0.0 and 1.0 "
                    "representing percentage, integer specifying exact number, or 'sqrt' for "
                    "square root of total features."
                )

            self.lambda_param = lambda_param
            self.drift_detection_method = (
                _extract_moa_drift_detector_CLI(ADWIN(delta=0.001))
                if drift_detection_method is None
                else _extract_moa_drift_detector_CLI(drift_detection_method)
            )
            self.warning_detection_method = (
                _extract_moa_drift_detector_CLI(ADWIN(delta=0.01))
                if warning_detection_method is None
                else _extract_moa_drift_detector_CLI(warning_detection_method)
            )
            self.disable_weighted_vote = disable_weighted_vote
            self.disable_drift_detection = disable_drift_detection
            self.disable_background_learner = disable_background_learner

            if number_of_jobs is None or number_of_jobs == 0 or number_of_jobs == 1:
                self.number_of_jobs = 1
            elif number_of_jobs < 0:
                self.number_of_jobs = os.cpu_count()
            else:
                self.number_of_jobs = int(min(number_of_jobs, os.cpu_count()))

            if minibatch_size is not None and minibatch_size > 1:
                self.minibatch_size = int(minibatch_size)
                moa_learner = _MOA_AdaptiveRandomForestMB()
                CLI = f"-l {self.base_learner} -s {self.ensemble_size} -o {self.m_features_mode} -m \
                                    {self.m_features_per_tree_size} -a {self.lambda_param} -x {self.drift_detection_method} -p \
                                    {self.warning_detection_method} {'-w' if self.disable_weighted_vote else ''} {'-u' if self.disable_drift_detection else ''}  \
                                    {'-q' if self.disable_background_learner else ''}\
                                    -c {self.number_of_jobs} -b {self.minibatch_size}"
            else:
                moa_learner = _MOA_AdaptiveRandomForest()
                CLI = f"-l {self.base_learner} -s {self.ensemble_size} -o {self.m_features_mode} -m \
                                    {self.m_features_per_tree_size} -a {self.lambda_param} -x {self.drift_detection_method} -p \
                                    {self.warning_detection_method} {'-w' if self.disable_weighted_vote else ''} {'-u' if self.disable_drift_detection else ''}  \
                                    {'-q' if self.disable_background_learner else ''}\
                                    -j {self.number_of_jobs}"

        super().__init__(
            schema=schema,
            CLI=CLI,
            random_seed=random_seed,
            moa_learner=moa_learner,
        )
