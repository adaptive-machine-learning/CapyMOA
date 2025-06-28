from __future__ import annotations

from capymoa.base import (
    MOAClassifier,
)
from capymoa.stream import Schema
from capymoa._utils import build_cli_str_from_mapping_and_locals

from moa.classifiers.meta import StreamingGradientBoostedTrees as _MOA_SGBT


class StreamingGradientBoostedTrees(MOAClassifier):
    """Streaming Gradient Boosted Trees.

    Streaming Gradient Boosted Trees (SGBT) [#0]_, which is trained using weighted
    squared loss elicited in XGBoost. SGBT exploits trees with a replacement strategy to
    detect and recover from drifts, thus enabling the ensemble to adapt without
    sacrificing the predictive performance.

    >>> from capymoa.classifier import StreamingGradientBoostedTrees
    >>> from capymoa.datasets import ElectricityTiny
    >>> from capymoa.evaluation import prequential_evaluation
    >>>
    >>> stream = ElectricityTiny()
    >>> classifier = StreamingGradientBoostedTrees(stream.get_schema())
    >>> results = prequential_evaluation(stream, classifier, max_instances=1000)
    >>> print(f"{results['cumulative'].accuracy():.1f}")
    86.3
    >>> learner = StreamingGradientBoostedTrees(
    ...     stream.get_schema(),
    ...     base_learner='meta.AdaptiveRandomForestRegressor -s 10',
    ...     boosting_iterations=10
    ... )
    >>> results = prequential_evaluation(stream, learner, max_instances=1000)
    >>> print(f"{results['cumulative'].accuracy():.1f}")
    86.8

    .. [#0] `Gradient boosted trees for evolving data streams. Nuwan Gunasekara,
             Bernhard Pfahringer, Heitor Murilo Gomes, Albert Bifet. Machine Learning,
             Springer, 2024. <https://doi.org/10.1007/s10994-024-06517-y>`_
    """

    def __init__(
        self,
        schema: Schema | None = None,
        random_seed: int = 0,
        base_learner="trees.FIMTDD -s VarianceReductionSplitCriterion -g 25 -c 0.05 -e -p",
        boosting_iterations: int = 100,
        percentage_of_features: int = 75,
        learning_rate=0.0125,
        disable_one_hot: bool = False,
        multiply_hessian_by: int = 1,
        skip_training: int = 1,
        use_squared_loss: bool = False,
    ):
        """Streaming Gradient Boosted Trees (SGBT) Classifier

        :param schema: The schema of the stream.
        :param random_seed: The random seed passed to the MOA learner.
        :param base_learner: The base learner to be trained. Default FIMTDD -s VarianceReductionSplitCriterion -g 25 -c 0.05 -e -p.
        :param boosting_iterations: The number of boosting iterations.
        :param percentage_of_features: The percentage of features to use.
        :param learning_rate: The learning rate.
        :param disable_one_hot: Whether to disable one-hot encoding for regressors that supports nominal attributes.
        :param multiply_hessian_by: The multiply hessian by this parameter to generate weights for multiple iterations.
        :param skip_training: Skip training of 1/skip_training instances. skip_training=1 means no skipping is performed (train on all instances).
        :param use_squared_loss: Whether to use squared loss for classification.
        """

        mapping = {
            "base_learner": "-l",
            "boosting_iterations": "-s",
            "percentage_of_features": "-m",
            "learning_rate": "-L",
            "disable_one_hot": "-H",
            "multiply_hessian_by": "-M",
            "skip_training": "-S",
            "use_squared_loss": "-K",
            "random_seed": "-r",
        }

        assert isinstance(base_learner, str), (
            "Only MOA CLI strings are supported for SGBT base_learner, at the moment."
        )

        config_str = build_cli_str_from_mapping_and_locals(mapping, locals())
        super(StreamingGradientBoostedTrees, self).__init__(
            moa_learner=_MOA_SGBT,
            schema=schema,
            CLI=config_str,
            random_seed=random_seed,
        )
