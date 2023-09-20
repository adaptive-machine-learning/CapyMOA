from prepare_jpype import start_jpype
import pandas as pd
start_jpype()

from moa.classifiers.meta import AdaptiveRandomForest as MOA_AdaptiveRandomForest
from moa.core import Utils

class AdaptiveRandomForest:
    def __init__(self, schema, tree_learner=None, ensemble_size=100, 
                 m_features_mode=None, m_features_per_tree_size=60, lambda_param=6.0,
                number_of_jobs=1, drift_detection_method=None, warning_detection_method=None,
                disable_weighted_vote=False, disable_drift_detection=False, disable_background_learner=False):
        # Initialize instance attributes with default values        
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

        self.moa_learner = MOA_AdaptiveRandomForest()
        # self.moa_learner.getOptions().setViaCLIString(f"-l {self.tree_learner} -s {self.ensemble_size}" -o {self.m_features_mode} -m {self.m_features_per_tree_size} -a {self.lambda} -j {self.number_of_jobs} -x {self.drift_detection_method} -p {self.warning_detection_method} {'-w' if self.disable_weighted_vote else ''} {'-u' if self.disable_drift_detection else ''}  {'-q' if self.disable_background_learner else ''}")
        self.moa_learner.getOptions().setViaCLIString(f"-l {self.tree_learner} -s {self.ensemble_size} -o {self.m_features_mode} -m {self.m_features_per_tree_size} -a {self.lambda_param} -j {self.number_of_jobs} -x {self.drift_detection_method} -p {self.warning_detection_method} {'-w' if self.disable_weighted_vote else ''} {'-u' if self.disable_drift_detection else ''}  {'-q' if self.disable_background_learner else ''}")
        # self.moa_learner.setRandomSeed(1)
        self.moa_learner.prepareForUse()
        self.moa_learner.resetLearningImpl()

        self.moa_learner.setModelContext(schema.getMoaHeader())

    def train(self, instance):
        self.moa_learner.trainOnInstance(instance)

    def predict(self, instance):
        return Utils.maxIndex(self.moa_learner.getVotesForInstance(instance))

    def predict_proba(self, instance):
        return self.moa_learner.getVotesForInstance(instance)