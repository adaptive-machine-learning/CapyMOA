# Library imports

from capymoa.base import MOARegressor, _extract_moa_learner_CLI
from ._arffimtdd import ARFFIMTDD


from moa.classifiers.meta import (
    AdaptiveRandomForestRegressor as MOA_AdaptiveRandomForestRegressor,
)


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

        super().__init__(
            schema=schema,
            CLI=CLI,
            random_seed=random_seed,
            moa_learner=self.moa_learner,
        )