from typing import Optional, Union

from capymoa.base import (
    MOARegressor,
)

from capymoa.splitcriteria import SplitCriterion, _split_criterion_to_cli_str
from capymoa.stream._stream import Schema
from moa.classifiers.trees import SelfOptimisingBaseTree as _MOA_SelfOptimisingBaseTree



class SOKNLBT(MOARegressor):
    """The base tree for Self-Optimising K Nearest Leaves as distribed by Sun. at el.

    SOKNLBT modifies the FIMT-DD algorithm to store information at the leaves that
    allows SOKNL to calculate the distance between a given instance and a leaf.

    See also :py:class:`capymoa.regressor.SOKNL` and py:class:`capymoa.regressor.FIMTDD`
    See :py:class:`capymoa.base.MOARegressor` for train and predict.

    Reference:

    `Sun, Yibin, Bernhard Pfahringer, Heitor Murilo Gomes, and Albert Bifet.
    "SOKNL: A novel way of integrating K-nearest neighbours with adaptive random forest regression for data streams."
    Data Mining and Knowledge Discovery 36, no. 5 (2022): 2006-2032.
    <https://researchcommons.waikato.ac.nz/server/api/core/bitstreams/f91959c0-1515-44c3-bd5f-737135ee3e48/content>`_

    Example usage:

    >>> from capymoa.datasets import Fried
        >>> from capymoa.regressor import SOKNLBT
        >>> from capymoa.evaluation import prequential_evaluation
    >>> stream = Fried()
    >>> schema = stream.get_schema()
    >>> learner = SOKNLBT(schema)
    >>> results = prequential_evaluation(stream, learner, max_instances=1000)
    >>> results["cumulative"].rmse()
    4.950050301515773
    """

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

