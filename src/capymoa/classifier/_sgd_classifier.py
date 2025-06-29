from typing import Optional, Literal
from capymoa.base import SKClassifier
from sklearn.linear_model import (
    SGDClassifier as _SKSGDClassifier,
)
from capymoa.stream._stream import Schema


class SGDClassifier(SKClassifier):
    """Streaming stochastic gradient descent classifier.

    This wraps :class:`~sklearn.linear_model.SGDClassifier` for
    ease of use in the streaming context. Some options are missing because
    they are not relevant in the streaming context. Furthermore, the learning rate
    is constant.

    >>> from capymoa.datasets import ElectricityTiny
    >>> from capymoa.classifier import PassiveAggressiveClassifier
    >>> from capymoa.evaluation import prequential_evaluation
    >>> stream = ElectricityTiny()
    >>> schema = stream.get_schema()
    >>> learner = SGDClassifier(schema)
    >>> results = prequential_evaluation(stream, learner, max_instances=1000)
    >>> results["cumulative"].accuracy()
    84.2
    """

    sklearner: _SKSGDClassifier
    """The underlying scikit-learn object"""

    def __init__(
        self,
        schema: Schema,
        loss: Literal[
            "hinge",
            "log_loss",
            "modified_huber",
            "squared_hinge",
            "perceptron",
            "squared_error",
            "huber",
            "epsilon_insensitive",
            "squared_epsilon_insensitive",
        ] = "hinge",
        penalty: Literal["l2", "l1", "easticnet"] = "l2",
        alpha: float = 0.0001,
        l1_ratio: float = 0.15,
        fit_intercept: bool = True,
        epsilon: float = 0.1,
        n_jobs: Optional[int] = None,
        learning_rate: Literal["constant", "optimal", "invscaling"] = "optimal",
        eta0: float = 0.0,
        random_seed: Optional[int] = None,
    ):
        """Construct stochastic gradient descent classifier.

        :param schema: Describes the datastream's structure.
        :param loss: The loss function to be used.
        :param penalty: The penalty (aka regularization term) to be used.
        :param alpha: Constant that multiplies the regularization term.
        :param l1_ratio: The Elastic Net mixing parameter, with ``0 <= l1_ratio <= 1``.
            ``l1_ratio=0`` corresponds to L2 penalty, ``l1_ratio=1`` to L1.
            Only used if ``penalty`` is 'elasticnet'.
            Values must be in the range ``[0.0, 1.0]``.
        :param fit_intercept: Whether the intercept (bias) should be estimated
            or not. If False, the data is assumed to be already centered.
        :param epsilon: Epsilon in the epsilon-insensitive loss functions; only
            if ``loss`` is 'huber', 'epsilon_insensitive', or
            'squared_epsilon_insensitive'. For 'huber', determines the threshold
            at which it becomes less important to get the prediction exactly right.
            For epsilon-insensitive, any differences between the current prediction
            and the correct label are ignored if they are less than this threshold.
        :param n_jobs: The number of CPUs to use to do the OVA (One Versus All, for
            multi-class problems) computation. Defaults to 1.
        :param learning_rate: The size of the gradient step.
        :param eta0: The initial learning rate for the 'constant', 'invscaling' or
            'adaptive' schedules. The default value is 0.0 as ``eta0`` is not used by
            the default schedule 'optimal'.
        :param class_weight: Weights associated with classes. If not given, all classes
            are supposed to have weight one.

            The "balanced" mode uses the values of y to automatically adjust
            weights inversely proportional to class frequencies in the input data
            as ``n_samples / (n_classes * np.bincount(y))``.
        :param random_seed: Seed for reproducibility.
        """

        super().__init__(
            _SKSGDClassifier(
                loss=loss,
                penalty=penalty,
                alpha=alpha,
                l1_ratio=l1_ratio,
                fit_intercept=fit_intercept,
                epsilon=epsilon,
                n_jobs=n_jobs,
                learning_rate=learning_rate,
                eta0=eta0,
                random_state=random_seed,
            ),
            schema,
            random_seed,
        )

    def __str__(self):
        return str("SGDClassifier")
