# Create the JVM and add the MOA jar to the classpath
from prepare_jpype import start_jpype

start_jpype()

# Library imports
from learners import MOARegressor, _get_moa_creation_CLI, _extract_moa_learner_CLI

# MOA/Java imports
from moa.classifiers.lazy import kNN as MOA_kNN
from moa.classifiers.meta import (
    AdaptiveRandomForestRegressor as MOA_AdaptiveRandomForestRegressor,
)


class KNNRegressor(MOARegressor):
    """
    The default number of neighbors (k) is set to 3 instead of 10 (as in MOA)
    """

    def __init__(
        self, schema=None, CLI=None, random_seed=1, k=3, median=False, window_size=1000
    ):
        # Important, should create the MOA object before invoking the super class __init__
        self.moa_learner = MOA_kNN()
        super().__init__(
            schema=schema,
            CLI=CLI,
            random_seed=random_seed,
            moa_learner=self.moa_learner,
        )

        # Initialize instance attributes with default values, CLI was not set.
        if self.CLI is None:
            self.k = k
            self.median = median
            self.window_size = window_size
            self.moa_learner.getOptions().setViaCLIString(
                f"-k {self.k} {'-m' if self.median else ''} -w \
             {self.window_size}"
            )
            self.moa_learner.prepareForUse()
            self.moa_learner.resetLearning()

    def __str__(self):
        # Overrides the default class name from MOA
        return "kNNRegressor"


########################
####### ENSEMBLES ######
########################


# TODO: replace the m_features_mode logic such that we can infer from m_features_per_tree_size, e.g. if value is double between 0.0 and 1.0 = percentage
class AdaptiveRandomForestRegressor(MOARegressor):
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
    ):
        # Important: must create the MOA object before invoking the super class __init__
        self.moa_learner = MOA_AdaptiveRandomForestRegressor()
        super().__init__(
            schema=schema,
            CLI=CLI,
            random_seed=random_seed,
            moa_learner=self.moa_learner,
        )

        # Initialize instance attributes with default values, CLI was not set.
        if self.CLI is None:
            self.tree_learner = (
                "(ARFFIMTDD -s VarianceReductionSplitCriterion -g 50 -c 0.01)"
                if tree_learner is None
                else tree_learner
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

            self.moa_learner.getOptions().setViaCLIString(
                f"-l {self.tree_learner} -s {self.ensemble_size} -o {self.m_features_mode} -m \
                {self.m_features_per_tree_size} -a {self.lambda_param} -x {self.drift_detection_method} -p \
                {self.warning_detection_method} {'-u' if self.disable_drift_detection else ''}  {'-q' if self.disable_background_learner else ''}"
            )
            self.moa_learner.prepareForUse()
            self.moa_learner.resetLearning()
