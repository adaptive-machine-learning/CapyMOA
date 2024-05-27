from capymoa.base import (
    MOAClassifier,
    _extract_moa_learner_CLI,
)

from moa.classifiers.meta import StreamingRandomPatches as _MOA_StreamingRandomPatches

class StreamingRandomPatches(MOAClassifier):
    
    
    def __init__(
        self,
        schema=None,
        CLI=None,
        random_seed=1,
        base_learner=None,
        ensemble_size=100,
        max_features=0.6,
        lambda_param=6.0,
        training_method=None,
        drift_detection_method=None,
        warning_detection_method=None,
        disable_weighted_vote=False,
        disable_drift_detection=False,
        disable_background_learner=False,
    ):
        """Construct an SRP classifier.

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
        :param lambda_param: The lambda parameter that controls the Poisson distribution for
            the online bagging simulation.
        :param drift_detection_method: The method used for drift detection.
        :param warning_detection_method: The method used for warning detection.
        :param disable_weighted_vote: Whether to disable weighted voting.
        :param disable_drift_detection: Whether to disable drift detection.
        :param disable_background_learner: Whether to disable background learning.
        """
        
        if CLI is None:
            self.base_learner = (
                "(trees.HoeffdingTree -g 50 -c 0.01)"
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
                raise ValueError("Invalid value for max_features. Valid options: float between 0.0 and 1.0 "
                                 "representing percentage, integer specifying exact number, or 'sqrt' for "
                                 "square root of total features.")

            self.lambda_param = lambda_param
            self.training_method = (
                "Random Patches"
                if training_method is None
                else training_method)
            self.drift_detection_method = (
                "(ADWINChangeDetector -a 1.0E-5)"
                if drift_detection_method is None
                else drift_detection_method
            )
            self.warning_detection_method = (
                "(ADWINChangeDetector -a 1.0E-4)"
                if warning_detection_method is None
                else warning_detection_method
            )
            self.disable_weighted_vote = disable_weighted_vote
            self.disable_drift_detection = disable_drift_detection
            self.disable_background_learner = disable_background_learner
            CLI = f"-l {self.base_learner} -s {self.ensemble_size} -o {self.m_features_mode} -m \
                {self.m_features_per_tree_size} -a {self.lambda_param} -x {self.drift_detection_method} -p \
                {self.warning_detection_method} {'-w' if self.disable_weighted_vote else ''} {'-u' if self.disable_drift_detection else ''}  \
                {'-q' if self.disable_background_learner else ''}"
        super().__init__(
            schema=schema,
            CLI=CLI,
            random_seed=random_seed,
            moa_learner=_MOA_StreamingRandomPatches(),
        )