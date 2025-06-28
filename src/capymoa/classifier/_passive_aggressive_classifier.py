from typing import Optional, Dict, Union, Literal
from capymoa.base import SKClassifier
from sklearn.linear_model import (
    PassiveAggressiveClassifier as _SKPassiveAggressiveClassifier,
)
from capymoa.stream._stream import Schema


class PassiveAggressiveClassifier(SKClassifier):
    """Streaming Passive Aggressive Classifier.

    Streaming Passive Aggressive Classifier [#0]_ is a classifier. This wraps
    :class:`~sklearn.linear_model.PassiveAggressiveClassifier` for ease of use in the streaming
    context. Some options are missing because they are not relevant in the streaming
    context.

    >>> from capymoa.classifier import PassiveAggressiveClassifier
    >>> from capymoa.datasets import ElectricityTiny
    >>> from capymoa.evaluation import prequential_evaluation
    >>>
    >>> stream = ElectricityTiny()
    >>> classifier = PassiveAggressiveClassifier(stream.get_schema())
    >>> results = prequential_evaluation(stream, classifier, max_instances=1000)
    >>> print(f"{results['cumulative'].accuracy():.1f}")
    84.3

    .. [#0] `Online Passive-Aggressive Algorithms K. Crammer, O. Dekel, J. Keshat, S.
             Shalev-Shwartz, Y. Singer - JMLR (2006)
             <http://jmlr.csail.mit.edu/papers/volume7/crammer06a/crammer06a.pdf>`_
    """

    sklearner: _SKPassiveAggressiveClassifier
    """The underlying scikit-learn object. See: :sklearn:`linear_model.PassiveAggressiveClassifier`"""

    def __init__(
        self,
        schema: Schema,
        max_step_size: float = 1.0,
        fit_intercept: bool = True,
        loss: str = "hinge",
        n_jobs: Optional[int] = None,
        class_weight: Union[Dict[int, float], None, Literal["balanced"]] = None,
        average: bool = False,
        random_seed=1,
    ):
        """Construct a passive aggressive classifier.

        :param schema: Stream schema
        :param max_step_size: Maximum step size (regularization).
        :param fit_intercept: Whether the intercept should be estimated or not.
            If False, the data is assumed to be already centered.
        :param loss: The loss function to be used: hinge: equivalent to PA-I in
            the reference paper. squared_hinge: equivalent to PA-II in the reference paper.
        :param n_jobs: The number of CPUs to use to do the OVA (One Versus All,
            for multi-class problems) computation. None means 1 unless in a
            ``joblib.parallel_backend`` context. -1 means using all processors.
        :param class_weight: Preset for the ``sklearner.class_weight`` fit parameter.

            Weights associated with classes. If not given, all classes are
            supposed to have weight one.

            The “balanced” mode uses the values of y to automatically adjust
            weights inversely proportional to class frequencies in the input
            data as ``n_samples / (n_classes * np.bincount(y))``.
        :param average: When set to True, computes the averaged SGD weights and
            stores the result in the ``sklearner.coef_`` attribute. If set to an int greater
            than 1, averaging will begin once the total number of samples
            seen reaches average. So ``average=10`` will begin averaging after
            seeing 10 samples.
        :param random_seed: Seed for the random number generator.
        """

        super().__init__(
            _SKPassiveAggressiveClassifier(
                C=max_step_size,
                fit_intercept=fit_intercept,
                early_stopping=False,
                shuffle=False,
                verbose=0,
                loss=loss,
                n_jobs=n_jobs,
                warm_start=False,
                class_weight=class_weight,
                average=average,
                random_state=random_seed,
            ),
            schema,
            random_seed,
        )

    def __str__(self):
        return str("PassiveAggressiveClassifier")
