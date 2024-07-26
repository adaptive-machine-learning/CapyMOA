from typing import Optional, Union

from capymoa.base import MOARegressor

from capymoa.splitcriteria import SplitCriterion, _split_criterion_to_cli_str
from capymoa.stream._stream import Schema
from moa.classifiers.trees import FIMTDD as _MOA_FIMTDD


class FIMTDD(MOARegressor):
    """Implementation of the FIMT-DD tree as described by Ikonomovska et al.

    Fast Incremental Model Tree with Drift Detection is the regression version
    for the famous Hoeffding Tree for data stream learning.

    FIMT-DD is implemented in MOA (Massive Online Analysis) and provides several
    parameters for customization.

    Reference:

    `Ikonomovska, Elena, João Gama, and Sašo Džeroski.
    Learning model trees from evolving data streams.
    Data mining and knowledge discovery 23.1 (2011): 128-168.
    <https://repositorio.inesctec.pt/server/api/core/bitstreams/a0802a15-84a2-493b-885b-a4f9fc4060b7/content>`_

    Example usage:

    >>> from capymoa.datasets import Fried
        >>> from capymoa.regressor import FIMTDD
        >>> from capymoa.evaluation import prequential_evaluation
    >>> stream = Fried()
    >>> schema = stream.get_schema()
    >>> learner = FIMTDD(schema)
    >>> results = prequential_evaluation(stream, learner, max_instances=1000)
    >>> results["cumulative"].rmse()
    7.363273627701553
    """

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
