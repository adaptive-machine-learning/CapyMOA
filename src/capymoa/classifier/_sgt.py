from typing import Literal

from moa.classifiers.trees import StochasticGradientTree as _StochasticGradientTree

from capymoa.base import MOAClassifier
from capymoa.stream import Schema


class StochasticGradientTree(MOAClassifier):
    """Stochastic Gradient Tree classifier.

    Stochastic Gradient Tree (SGT) [#f1]_ is an incremental decision tree that learns
    using stochastic gradient information as its source of supervision, rather than
    a heuristic such as information gain. Instead of using soft splits or rebuilding
    a new tree for every update, as prior gradient-based tree learners did in the
    batch setting, SGT accumulates per-node gradient and Hessian statistics online
    and uses them to make hard splitting decisions incrementally. Because splitting
    is driven only by the loss function's gradients and Hessians, the same algorithm
    can be applied to classification, regression, or multi-instance learning simply
    by changing the loss function.

    >>> from capymoa.classifier import StochasticGradientTree
    >>> from capymoa.datasets import ElectricityTiny
    >>> from capymoa.evaluation import prequential_evaluation
    >>>
    >>> stream = ElectricityTiny()
    >>> classifier = StochasticGradientTree(stream.get_schema())
    >>> results = prequential_evaluation(stream, classifier, max_instances=1000)
    >>> print(f"{results['cumulative'].accuracy():.1f}")
    50.6

    .. [#f1] Gouk, Henry, Bernhard Pfahringer, and Eibe Frank. "Stochastic Gradient
             Trees." Proceedings of The 11th Asian Conference on Machine Learning
             (ACML 2019). PMLR 101, 2019, pp. 1094-1109.
    """

    def __init__(
        self,
        schema: Schema,
        grace_period: int = 200,
        lambda_: float = 0.1,
        warm_start: int = 1000,
        confidence: float = 1e-06,
        split_test: Literal["TTest"] = "TTest",
        disable_resplits: bool = False,
    ) -> None:
        """Construct StochasticGradientTree classifier.

        :param grace_period: The number of instances a leaf should observe between
            split attempts.
        :param lambda_: Regularization parameter lambda.
        :param warm_start: Number of instances to use for fitting the discretizers.
        :param confidence: The level of confidence required that a split candidate is
            an improvement before the split is actually performed.
        :param split_test: Which type of hypothesis test to use for determining when
            to split.

            * ``TTest``: Use a t-Test for checking statistical significance.
        :param disable_resplits: Disable node resplitting.
        """

        cli = []
        cli += [f"-G {grace_period}"]
        cli += [f"-L {lambda_}"]
        cli += [f"-W {warm_start}"]
        cli += [f"-C {confidence}"]
        cli += [f"-T '{split_test}'"]
        cli += ["-R"] if disable_resplits else []

        super().__init__(
            moa_learner=_StochasticGradientTree, schema=schema, CLI=" ".join(cli)
        )
