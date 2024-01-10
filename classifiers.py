# Create the JVM and add the MOA jar to the classpath
from prepare_jpype import start_jpype

start_jpype()


# Library imports
from learners import (
    MOAClassifier,
    MOARegressor,
    _get_moa_creation_CLI,
    _extract_moa_learner_CLI,
)

# MOA/Java imports
from moa.classifiers import Classifier
from moa.classifiers.meta import AdaptiveRandomForest as MOA_AdaptiveRandomForest
from moa.classifiers.meta import OzaBag as MOA_OzaBag
from moa.classifiers.meta import (
    AdaptiveRandomForestRegressor as MOA_AdaptiveRandomForestRegressor,
)


# TODO: replace the m_features_mode logic such that we can infer from m_features_per_tree_size, e.g. if value is double between 0.0 and 1.0 = percentage
class AdaptiveRandomForest(MOAClassifier):
    def __init__(
        self,
        schema=None,
        CLI=None,
        random_seed=1,
        base_learner=None,
        ensemble_size=100,
        max_features=0.6,
        lambda_param=6.0,  # m_features_mode=None, m_features_per_tree_size=60,
        number_of_jobs=1,
        drift_detection_method=None,
        warning_detection_method=None,
        disable_weighted_vote=False,
        disable_drift_detection=False,
        disable_background_learner=False,
    ):
        # Initialize instance attributes with default values, if the CLI was not set.
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
                # Handle other cases or raise an exception if needed
                raise ValueError("Invalid value for max_features")

            # self.m_features_mode = "(Percentage (M * (m / 100)))" if m_features_mode is None else m_features_mode
            # self.m_features_per_tree_size = m_features_per_tree_size
            self.lambda_param = lambda_param
            self.number_of_jobs = number_of_jobs
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
            self.disable_weighted_vote = disable_weighted_vote
            self.disable_drift_detection = disable_drift_detection
            self.disable_background_learner = disable_background_learner
            CLI = f"-l {self.base_learner} -s {self.ensemble_size} -o {self.m_features_mode} -m \
                {self.m_features_per_tree_size} -a {self.lambda_param} -j {self.number_of_jobs} -x {self.drift_detection_method} -p \
                {self.warning_detection_method} {'-w' if self.disable_weighted_vote else ''} {'-u' if self.disable_drift_detection else ''}  \
                {'-q' if self.disable_background_learner else ''}"
        super().__init__(
            schema=schema,
            CLI=CLI,
            random_seed=random_seed,
            moa_learner=MOA_AdaptiveRandomForest(),
        )


class OnlineBagging(MOAClassifier):
    def __init__(
        self, schema=None, CLI=None, random_seed=1, base_learner=None, ensemble_size=100
    ):
        # This method basically configures the CLI, object creation is delegated to MOAClassifier (the super class, through super().__init___()))
        # Initialize instance attributes with default values, if the CLI was not set.
        if CLI is None:
            self.base_learner = (
                "trees.HoeffdingTree"
                if base_learner is None
                else _extract_moa_learner_CLI(base_learner)
            )
            self.ensemble_size = ensemble_size
            CLI = f"-l {self.base_learner} -s {self.ensemble_size}"

        super().__init__(
            schema=schema, CLI=CLI, random_seed=random_seed, moa_learner=MOA_OzaBag()
        )

    def __str__(self):
        # Overrides the default class name from MOA (OzaBag)
        return "OnlineBagging"
