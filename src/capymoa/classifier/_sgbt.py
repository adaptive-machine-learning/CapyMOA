from __future__ import annotations

import jpype
from moa.classifiers.meta import StreamingGradientBoostedTrees as _MOA_SGBT

from capymoa.base import MOAClassifier, MOARegressor
from capymoa.base._seed import resolve_seed
from capymoa.stream import Schema


class StreamingGradientBoostedTrees(MOAClassifier):
    """Streaming Gradient Boosted Trees.

    Streaming Gradient Boosted Trees (SGBT) [#0]_, which is trained using weighted
    squared loss elicited in XGBoost. SGBT exploits trees with a replacement strategy to
    detect and recover from drifts, thus enabling the ensemble to adapt without
    sacrificing the predictive performance.

    >>> from capymoa.classifier import StreamingGradientBoostedTrees
    >>> from capymoa.datasets import ElectricityTiny
    >>> from capymoa.classifier import evaluate_classifier
    >>>
    >>> stream = ElectricityTiny()
    >>> classifier = StreamingGradientBoostedTrees(stream.get_schema(), random_seed=0)
    >>> results = evaluate_classifier(stream, classifier, max_instances=1000)
    >>> print(f"{results['cumulative']['accuracy']:.1f}")
    86.3
    >>> learner = StreamingGradientBoostedTrees(
    ...     stream.get_schema(),
    ...     base_learner='meta.AdaptiveRandomForestRegressor -s 10',
    ...     boosting_iterations=10,
    ...     random_seed=0,
    ... )
    >>> results = evaluate_classifier(stream, learner, max_instances=1000)
    >>> print(f"{results['cumulative']['accuracy']:.1f}")
    86.8

    .. [#0] `Gradient boosted trees for evolving data streams. Nuwan Gunasekara,
             Bernhard Pfahringer, Heitor Murilo Gomes, Albert Bifet. Machine Learning,
             Springer, 2024. <https://doi.org/10.1007/s10994-024-06517-y>`_
    """

    def __init__(
        self,
        schema: Schema | None = None,
        random_seed: int | None = None,
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
        :param base_learner: The base learner to be trained. A CLI string, a CapyMOA
            MOA learner, or a MOA Java object. Python learners are not supported.
            Default FIMTDD -s VarianceReductionSplitCriterion -g 25 -c 0.05 -e -p.
        :param boosting_iterations: The number of boosting iterations.
        :param percentage_of_features: The percentage of features to use.
        :param learning_rate: The learning rate.
        :param disable_one_hot: Whether to disable one-hot encoding for regressors that supports nominal attributes.
        :param multiply_hessian_by: The multiply hessian by this parameter to generate weights for multiple iterations.
        :param skip_training: Skip training of 1/skip_training instances. skip_training=1 means no skipping is performed (train on all instances).
        :param use_squared_loss: Whether to use squared loss for classification.
        """

        random_seed = resolve_seed(random_seed)
        if not isinstance(
            base_learner, (str, MOAClassifier, MOARegressor, jpype.JObject)
        ):
            # MOA copies the base learner with Java serialization, which the
            # proxies for Python learners do not support.
            raise TypeError("SGBT needs a MOA base learner, not a Python learner.")

        options = {
            "baseLearner": base_learner,
            "numberOfboostingIterations": boosting_iterations,
            "percentageOfAttributesForEachBoostingIteration": percentage_of_features,
            "learningRate": learning_rate,
            "disableOneHotEncoding": disable_one_hot,
            "multipleIterationByCeilingOfHessianTimesM": multiply_hessian_by,
            "randomlySkip1SthOfInstancesAtTraining": skip_training,
            "useSquaredLossForClassification": use_squared_loss,
            "randomSeed": random_seed,
        }
        super().__init__(
            moa_learner=_MOA_SGBT,
            schema=schema,
            random_seed=random_seed,
            options=options,
        )
