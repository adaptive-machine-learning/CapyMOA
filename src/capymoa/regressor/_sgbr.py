from __future__ import annotations

import jpype
from moa.classifiers.meta import StreamingGradientBoostedRegression as _MOA_SGBR

from capymoa.base import MOAClassifier, MOARegressor
from capymoa.base._seed import resolve_seed
from capymoa.stream import Schema


class StreamingGradientBoostedRegression(MOARegressor):
    """Streaming Gradient Boosted Regression.

    Streaming Gradient Boosted Regression (SGBR) [#0]_, was developed to adapt gradient boosting for streaming regression
    using Streaming Gradient Boosted Trees (SGBT). A variant called SGB(Oza), which uses OzaBag bagging regressors as
    base learners, outperforms existing state-of-the-art methods in both accuracy and efficiency across various drift
    scenarios.

    >>> from capymoa.datasets import Fried
        >>> from capymoa.regressor import StreamingGradientBoostedRegression
        >>> from capymoa.regressor import evaluate_regressor
    >>> stream = Fried()
    >>> schema = stream.get_schema()
    >>> learner = StreamingGradientBoostedRegression(schema, random_seed=0)
    >>> results = evaluate_regressor(stream, learner, max_instances=1000)
    >>> round(results["cumulative"]["r2"], 2)
    0.61

    .. [#0] `Gradient boosted bagging for evolving data stream regression. Nuwan Gunasekara,
             Bernhard Pfahringer, Heitor Murilo Gomes, Albert Bifet. Data Mining and Knowledge Discovery,
             Springer, 2025. <https://doi.org/10.1007/s10618-025-01147-x>`_
    """

    def __init__(
        self,
        schema: Schema | None = None,
        random_seed: int | None = None,
        base_learner="meta.OzaBag -s 10 -l (trees.FIMTDD -s VarianceReductionSplitCriterion -g 50 -c 0.01 -e)",
        boosting_iterations: int = 10,
        percentage_of_features: int = 75,
        learning_rate=1.0,
        disable_one_hot: bool = False,
        multiply_hessian_by: int = 1,
        skip_training: int = 1,
        use_squared_loss: bool = False,
    ):
        """Streaming Gradient Boosted Regression (SGBR) Regressor

        :param schema: The schema of the stream.
        :param random_seed: The random seed passed to the MOA learner.
        :param base_learner: The base learner to be trained. Default meta.OzaBag -s 10 -l (trees.FIMTDD -s VarianceReductionSplitCriterion -g 50 -c 0.01 -e).
        :param boosting_iterations: The number of boosting iterations. Default 10.
        :param percentage_of_features: The percentage of features to use.
        :param learning_rate: The learning rate. Default 1.0.
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
            raise TypeError("SGBR needs a MOA base learner, not a Python learner.")

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
            moa_learner=_MOA_SGBR(),
            schema=schema,
            random_seed=random_seed,
            options=options,
        )
