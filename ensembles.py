# Create the JVM and add the MOA jar to the classpath
from prepare_jpype import start_jpype
start_jpype()

# import pandas as pd

# Library imports
from MOALearners import MOAClassifier, MOARegressor

# MOA/Java imports
from moa.classifiers.meta import AdaptiveRandomForest as MOA_AdaptiveRandomForest
from moa.classifiers.meta import OzaBag as MOA_OzaBag
from moa.classifiers.meta import AdaptiveRandomForestRegressor as MOA_AdaptiveRandomForestRegressor


# TODO: replace the m_features_mode logic such that we can infer from m_features_per_tree_size, e.g. if value is double between 0.0 and 1.0 = percentage
class AdaptiveRandomForest(MOAClassifier):
    def __init__(self, schema=None, CLI=None, random_seed=1, tree_learner=None, ensemble_size=100, 
                 m_features_mode=None, m_features_per_tree_size=60, lambda_param=6.0,
                number_of_jobs=1, drift_detection_method=None, warning_detection_method=None,
                disable_weighted_vote=False, disable_drift_detection=False, disable_background_learner=False):
        
        # Important: must create the MOA object before invoking the super class __init__
        self.moa_learner = MOA_AdaptiveRandomForest()
        super().__init__(schema=schema, CLI=CLI, random_seed=random_seed, moa_learner=self.moa_learner)

        # Initialize instance attributes with default values, CLI was not set. 
        if self.CLI is None:
            self.tree_learner = "(ARFHoeffdingTree -e 2000000 -g 50 -c 0.01)" if tree_learner is None else tree_learner
            self.ensemble_size = ensemble_size
            self.m_features_mode = "(Percentage (M * (m / 100)))" if m_features_mode is None else m_features_mode
            self.m_features_per_tree_size = m_features_per_tree_size
            self.lambda_param = lambda_param
            self.number_of_jobs = number_of_jobs
            self.drift_detection_method = "(ADWINChangeDetector -a 1.0E-3)" if drift_detection_method is None else drift_detection_method
            self.warning_detection_method = "(ADWINChangeDetector -a 1.0E-2)" if warning_detection_method is None else warning_detection_method
            self.disable_weighted_vote = disable_weighted_vote
            self.disable_drift_detection = disable_drift_detection
            self.disable_background_learner = disable_background_learner
            self.moa_learner.getOptions().setViaCLIString(f"-l {self.tree_learner} -s {self.ensemble_size} -o {self.m_features_mode} -m \
                {self.m_features_per_tree_size} -a {self.lambda_param} -j {self.number_of_jobs} -x {self.drift_detection_method} -p \
                {self.warning_detection_method} {'-w' if self.disable_weighted_vote else ''} {'-u' if self.disable_drift_detection else ''}  \
                {'-q' if self.disable_background_learner else ''}")
            self.moa_learner.prepareForUse()
            self.moa_learner.resetLearning()


class OnlineBagging(MOAClassifier):
    def __init__(self, schema=None, CLI=None, random_seed=1, base_learner=None, ensemble_size=100):
        
        # Important, should create the MOA object before invoking the super class __init__
        self.moa_learner = MOA_OzaBag()
        super().__init__(schema=schema, CLI=CLI, random_seed=random_seed, moa_learner=self.moa_learner)

        # Initialize instance attributes with default values, CLI was not set. 
        if self.CLI is None:
            self.base_learner = "trees.HoeffdingTree" if base_learner is None else base_learner
            self.ensemble_size = ensemble_size
            self.moa_learner.getOptions().setViaCLIString(f"-l {self.base_learner} -s {self.ensemble_size}")
            self.moa_learner.prepareForUse()
            self.moa_learner.resetLearning()

    def __str__(self):
        # Overrides the default class name from MOA (OzaBag)
        return 'OnlineBagging'


########################
###### REGRESSORS ######
########################


# TODO: replace the m_features_mode logic such that we can infer from m_features_per_tree_size, e.g. if value is double between 0.0 and 1.0 = percentage
class AdaptiveRandomForestRegressor(MOARegressor):
    def __init__(self, schema=None, CLI=None, random_seed=1, tree_learner=None, ensemble_size=100, 
                 m_features_mode=None, m_features_per_tree_size=60, lambda_param=6.0,
                drift_detection_method=None, warning_detection_method=None,
                disable_drift_detection=False, disable_background_learner=False):
        
        # Important: must create the MOA object before invoking the super class __init__
        self.moa_learner = MOA_AdaptiveRandomForestRegressor()
        super().__init__(schema=schema, CLI=CLI, random_seed=random_seed, moa_learner=self.moa_learner)

        # Initialize instance attributes with default values, CLI was not set. 
        if self.CLI is None:
            self.tree_learner = "(ARFFIMTDD -s VarianceReductionSplitCriterion -g 50 -c 0.01)" if tree_learner is None else tree_learner
            self.ensemble_size = ensemble_size
            self.m_features_mode = "(Percentage (M * (m / 100)))" if m_features_mode is None else m_features_mode
            self.m_features_per_tree_size = m_features_per_tree_size
            self.lambda_param = lambda_param
            self.drift_detection_method = "(ADWINChangeDetector -a 1.0E-3)" if drift_detection_method is None else drift_detection_method
            self.warning_detection_method = "(ADWINChangeDetector -a 1.0E-2)" if warning_detection_method is None else warning_detection_method
            self.disable_drift_detection = disable_drift_detection
            self.disable_background_learner = disable_background_learner

            self.moa_learner.getOptions().setViaCLIString(f"-l {self.tree_learner} -s {self.ensemble_size} -o {self.m_features_mode} -m \
                {self.m_features_per_tree_size} -a {self.lambda_param} -x {self.drift_detection_method} -p \
                {self.warning_detection_method} {'-u' if self.disable_drift_detection else ''}  {'-q' if self.disable_background_learner else ''}")
            self.moa_learner.prepareForUse()
            self.moa_learner.resetLearning()