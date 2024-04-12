# Library imports
from typing import Optional, Union

from capymoa.learner.learners import (
    MOARegressor,
)

from capymoa.learner.splitcriteria import SplitCriterion, _split_criterion_to_cli_str
from capymoa.stream.stream import Schema
from moa.classifiers.lazy import kNN as MOA_kNN
from moa.classifiers.meta import (
    AdaptiveRandomForestRegressor as MOA_AdaptiveRandomForestRegressor,
    SelfOptimisingKNearestLeaves as MOA_SOKNL,
)
from moa.classifiers.trees import (
    FIMTDD as _MOA_FIMTDD,
    ARFFIMTDD as _MOA_ARFFIMTDD,
    ORTO as _MOA_ORTO,
    SelfOptimisingBaseTree as _MOA_SelfOptimisingBaseTree,
)


########################
######### TREES ########
########################
class FIMTDD(MOARegressor):
    """Implementation of the FIMT-DD tree as described by Ikonomovska et al."""

    def __init__(
        self,
        schema: Schema,
        split_criterion: Union[SplitCriterion, str] = "VarianceReductionSplitCriterion",
        grace_period: int = 200,
        split_confidence: float = 1.0e-7,
        tie_threshold: float = 0.05,
        page_hinckley_alpha: float = 0.005,
        page_hinckley_threshold: int = 50,
        alternate_tree_fading_factor: float = 0.995,
        alternate_tree_t_min: int = 150,
        alternate_tree_time: int = 1500,
        regression_tree: bool = False,
        learning_ratio: float = 0.02,
        learning_ratio_decay_factor: float = 0.001,
        learning_ratio_const: bool = False,
        random_seed: Optional[int] = None,
    ) -> None:
        """
        Construct FIMTDD.

        :param split_criterion: Split criterion to use.
        :param grace_period: Number of instances a leaf should observe between split attempts.
        :param split_confidence: Allowed error in split decision, values close to 0 will take long to decide.
        :param tie_threshold: Threshold below which a split will be forced to break ties.
        :param page_hinckley_alpha: Alpha value to use in the Page Hinckley change detection tests.
        :param page_hinckley_threshold: Threshold value used in the Page Hinckley change detection tests.
        :param alternate_tree_fading_factor: Fading factor used to decide if an alternate tree should replace an original.
        :param alternate_tree_t_min: Tmin value used to decide if an alternate tree should replace an original.
        :param alternate_tree_time: The number of instances used to decide if an alternate tree should be discarded.
        :param regression_tree: Build a regression tree instead of a model tree.
        :param learning_ratio: Learning ratio to used for training the Perceptrons in the leaves.
        :param learning_ratio_decay_factor: Learning rate decay factor (not used when learning rate is constant).
        :param learning_ratio_const: Keep learning rate constant instead of decaying.
        """
        cli = []

        cli.append(f"-s ({_split_criterion_to_cli_str(split_criterion)})")
        cli.append(f"-g {grace_period}")
        cli.append(f"-c {split_confidence}")
        cli.append(f"-t {tie_threshold}")
        cli.append(f"-a {page_hinckley_alpha}")
        cli.append(f"-h {page_hinckley_threshold}")
        cli.append(f"-f {alternate_tree_fading_factor}")
        cli.append(f"-y {alternate_tree_t_min}")
        cli.append(f"-u {alternate_tree_time}")
        cli.append("-e") if regression_tree else None
        cli.append(f"-l {learning_ratio}")
        cli.append(f"-d {learning_ratio_decay_factor}")
        cli.append("-p") if learning_ratio_const else None

        self.moa_learner = _MOA_FIMTDD()

        super().__init__(
            schema=schema,
            CLI=" ".join(cli),
            random_seed=random_seed,
            moa_learner=self.moa_learner,
        )


class ARFFIMTDD(MOARegressor):
    """Modified Fast Incremental Model Tree with Drift Detection for basic
    learner for ARF-Regas described by Ikonomovska et al."""

    def __init__(
        self,
        schema: Schema,
        subspace_size_size: int = 2,
        split_criterion: Union[SplitCriterion, str] = "VarianceReductionSplitCriterion",
        grace_period: int = 200,
        split_confidence: float = 1.0e-7,
        tie_threshold: float = 0.05,
        page_hinckley_alpha: float = 0.005,
        page_hinckley_threshold: int = 50,
        alternate_tree_fading_factor: float = 0.995,
        alternate_tree_t_min: int = 150,
        alternate_tree_time: int = 1500,
        learning_ratio: float = 0.02,
        learning_ratio_decay_factor: float = 0.001,
        learning_ratio_const: bool = False,
        random_seed: Optional[int] = None,
    ) -> None:
        """
        Construct ARFFIMTDD.

        :param subspace_size_size: Number of features per subset for each node split. Negative values = #features - k
        :param split_criterion: Split criterion to use.
        :param grace_period: Number of instances a leaf should observe between split attempts.
        :param split_confidence: Allowed error in split decision, values close to 0 will take long to decide.
        :param tie_threshold: Threshold below which a split will be forced to break ties.
        :param page_hinckley_alpha: Alpha value to use in the Page Hinckley change detection tests.
        :param page_hinckley_threshold: Threshold value used in the Page Hinckley change detection tests.
        :param alternate_tree_fading_factor: Fading factor used to decide if an alternate tree should replace an original.
        :param alternate_tree_t_min: Tmin value used to decide if an alternate tree should replace an original.
        :param alternate_tree_time: The number of instances used to decide if an alternate tree should be discarded.
        :param learning_ratio: Learning ratio to used for training the Perceptrons in the leaves.
        :param learning_ratio_decay_factor: Learning rate decay factor (not used when learning rate is constant).
        :param learning_ratio_const: Keep learning rate constant instead of decaying.
        """
        cli = []

        cli.append(f"-k {subspace_size_size}")
        cli.append(f"-s ({_split_criterion_to_cli_str(split_criterion)})")
        cli.append(f"-g {grace_period}")
        cli.append(f"-c {split_confidence}")
        cli.append(f"-t {tie_threshold}")
        cli.append(f"-a {page_hinckley_alpha}")
        cli.append(f"-h {page_hinckley_threshold}")
        cli.append(f"-f {alternate_tree_fading_factor}")
        cli.append(f"-y {alternate_tree_t_min}")
        cli.append(f"-u {alternate_tree_time}")
        cli.append(f"-l {learning_ratio}")
        cli.append(f"-d {learning_ratio_decay_factor}")
        cli.append("-p") if learning_ratio_const else None

        self.moa_learner = _MOA_ARFFIMTDD()

        super().__init__(
            schema=schema,
            CLI=" ".join(cli),
            random_seed=random_seed,
            moa_learner=self.moa_learner,
        )


class ORTO(MOARegressor):
    """Implementation of the ORTO tree as described by Ikonomovska et al."""

    def __init__(
        self,
        schema: Schema,
        max_trees: int = 10,
        max_option_level: int = 10,
        option_decay_factor: float = 0.9,
        option_fading_factor: float = 0.9995,
        split_criterion: Union[SplitCriterion, str] = "VarianceReductionSplitCriterion",
        grace_period: int = 200,
        split_confidence: float = 1.0e-7,
        tie_threshold: float = 0.05,
        page_hinckley_alpha: float = 0.005,
        page_hinckley_threshold: int = 50,
        alternate_tree_fading_factor: float = 0.995,
        alternate_tree_t_min: int = 150,
        alternate_tree_time: int = 1500,
        regression_tree: bool = False,
        learning_ratio: float = 0.02,
        learning_ratio_decay_factor: float = 0.001,
        learning_ratio_const: bool = False,
        random_seed: Optional[int] = None,
    ) -> None:
        """
        Construct ORTO.

        :param max_trees: The maximum number of trees contained in the option tree.
        :param max_option_level: The maximal depth at which option nodes can be created.
        :param option_decay_factor: The option decay factor that determines how many options can be selected at a given level.
        :param option_fading_factor: The fading factor used for comparing subtrees of an option node.
        :param split_criterion: Split criterion to use.
        :param grace_period: Number of instances a leaf should observe between split attempts.
        :param split_confidence: Allowed error in split decision, values close to 0 will take long to decide.
        :param tie_threshold: Threshold below which a split will be forced to break ties.
        :param page_hinckley_alpha: Alpha value to use in the Page Hinckley change detection tests.
        :param page_hinckley_threshold: Threshold value used in the Page Hinckley change detection tests.
        :param alternate_tree_fading_factor: Fading factor used to decide if an alternate tree should replace an original.
        :param alternate_tree_t_min: Tmin value used to decide if an alternate tree should replace an original.
        :param alternate_tree_time: The number of instances used to decide if an alternate tree should be discarded.
        :param regression_tree: Build a regression tree instead of a model tree.
        :param learning_ratio: Learning ratio to used for training the Perceptrons in the leaves.
        :param learning_ratio_decay_factor: Learning rate decay factor (not used when learning rate is constant).
        :param learning_ratio_const: Keep learning rate constant instead of decaying.
        """
        cli = []

        cli.append(f"-m {max_trees}")
        cli.append(f"-x {max_option_level}")
        cli.append(f"-z {option_decay_factor}")
        cli.append(f"-q {option_fading_factor}")
        cli.append(f"-s ({_split_criterion_to_cli_str(split_criterion)})")
        cli.append(f"-g {grace_period}")
        cli.append(f"-c {split_confidence}")
        cli.append(f"-t {tie_threshold}")
        cli.append(f"-a {page_hinckley_alpha}")
        cli.append(f"-h {page_hinckley_threshold}")
        cli.append(f"-f {alternate_tree_fading_factor}")
        cli.append(f"-y {alternate_tree_t_min}")
        cli.append(f"-u {alternate_tree_time}")
        cli.append("-e") if regression_tree else None
        cli.append(f"-l {learning_ratio}")
        cli.append(f"-d {learning_ratio_decay_factor}")
        cli.append("-p") if learning_ratio_const else None

        self.moa_learner = _MOA_ORTO()

        super().__init__(
            schema=schema,
            CLI=" ".join(cli),
            random_seed=random_seed,
            moa_learner=self.moa_learner,
        )


class SOKNLBT(MOARegressor):
    """Implementation of the FIMT-DD tree as described by Ikonomovska et al."""

    def __init__(
        self,
        schema: Schema,
        subspace_size_size: int = 2,
        split_criterion: Union[SplitCriterion, str] = "VarianceReductionSplitCriterion",
        grace_period: int = 200,
        split_confidence: float = 1.0e-7,
        tie_threshold: float = 0.05,
        page_hinckley_alpha: float = 0.005,
        page_hinckley_threshold: int = 50,
        alternate_tree_fading_factor: float = 0.995,
        alternate_tree_t_min: int = 150,
        alternate_tree_time: int = 1500,
        learning_ratio: float = 0.02,
        learning_ratio_decay_factor: float = 0.001,
        learning_ratio_const: bool = False,
        random_seed: Optional[int] = None,
    ) -> None:
        """
        Construct SelfOptimisingBaseTree.

        :param subspace_size_size: Number of features per subset for each node split. Negative values = #features - k
        :param split_criterion: Split criterion to use.
        :param grace_period: Number of instances a leaf should observe between split attempts.
        :param split_confidence: Allowed error in split decision, values close to 0 will take long to decide.
        :param tie_threshold: Threshold below which a split will be forced to break ties.
        :param page_hinckley_alpha: Alpha value to use in the Page Hinckley change detection tests.
        :param page_hinckley_threshold: Threshold value used in the Page Hinckley change detection tests.
        :param alternate_tree_fading_factor: Fading factor used to decide if an alternate tree should replace an original.
        :param alternate_tree_t_min: Tmin value used to decide if an alternate tree should replace an original.
        :param alternate_tree_time: The number of instances used to decide if an alternate tree should be discarded.
        :param learning_ratio: Learning ratio to used for training the Perceptrons in the leaves.
        :param learning_ratio_decay_factor: Learning rate decay factor (not used when learning rate is constant).
        :param learning_ratio_const: Keep learning rate constant instead of decaying.
        """
        cli = []

        cli.append(f"-k {subspace_size_size}")
        cli.append(f"-s ({_split_criterion_to_cli_str(split_criterion)})")
        cli.append(f"-g {grace_period}")
        cli.append(f"-c {split_confidence}")
        cli.append(f"-t {tie_threshold}")
        cli.append(f"-a {page_hinckley_alpha}")
        cli.append(f"-h {page_hinckley_threshold}")
        cli.append(f"-f {alternate_tree_fading_factor}")
        cli.append(f"-y {alternate_tree_t_min}")
        cli.append(f"-u {alternate_tree_time}")
        cli.append(f"-l {learning_ratio}")
        cli.append(f"-d {learning_ratio_decay_factor}")
        cli.append("-p") if learning_ratio_const else None

        self.moa_learner = _MOA_SelfOptimisingBaseTree()

        super().__init__(
            schema=schema,
            CLI=" ".join(cli),
            random_seed=random_seed,
            moa_learner=self.moa_learner,
        )


########################
######### LAZY #########
########################


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
                ARFFIMTDD(schema, grace_period=50, split_confidence=0.01)
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


class SOKNL(MOARegressor):
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
        self_optimising=True,
        k_value=10,
    ):
        # Important: must create the MOA object before invoking the super class __init__
        self.moa_learner = MOA_SOKNL()
        super().__init__(
            schema=schema,
            CLI=CLI,
            random_seed=random_seed,
            moa_learner=self.moa_learner,
        )

        # Initialize instance attributes with default values, CLI was not set.
        if self.CLI is None:
            self.tree_learner = (
                # "(SelfOptimisingBaseTree -s VarianceReductionSplitCriterion -g 50 -c 0.01)"
                SOKNLBT(schema, grace_period=50, split_confidence=0.01)
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

            self.self_optimising = self_optimising
            self.k_value = k_value

            self.moa_learner.getOptions().setViaCLIString(
                f"-l {self.tree_learner} -s {self.ensemble_size} {'-f' if self.self_optimising else ''} -k {self.k_value} -o {self.m_features_mode} -m \
                {self.m_features_per_tree_size} -a {self.lambda_param} -x {self.drift_detection_method} -p \
                {self.warning_detection_method} {'-u' if self.disable_drift_detection else ''}  {'-q' if self.disable_background_learner else ''}"
            )
            self.moa_learner.prepareForUse()
            self.moa_learner.resetLearning()
