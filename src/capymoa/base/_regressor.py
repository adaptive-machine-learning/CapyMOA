from abc import ABC, abstractmethod

import numpy as np
from numpy.typing import NDArray
from sklearn.base import RegressorMixin as _SKRegressorMixin

from capymoa.base._batch import _BatchBuilder
from capymoa.instance import Instance, RegressionInstance
from capymoa.stream._stream import Schema
from capymoa.type_alias import TargetValue


class Regressor(ABC):
    def __init__(self, schema=None, random_seed=1):
        self.random_seed = random_seed
        self.schema = schema

    def __str__(self):
        pass

    @abstractmethod
    def train(self, instance: RegressionInstance):
        pass

    @abstractmethod
    def predict(self, instance: RegressionInstance) -> TargetValue:
        pass


class BatchRegressor(Regressor):
    """Base class for batch trained regression algorithms.

    >>> class MyBatchRegressor(BatchRegressor):
    ...
    ...     def batch_train(self, x, y):
    ...         with np.printoptions(precision=2):
    ...             print(x)
    ...             print(y)
    ...             print()
    ...
    ...     def predict(self, instance):
    ...         return 0.0
    ...
    >>> from capymoa.datasets import FriedTiny
    >>> print("downloading stream"); stream = FriedTiny()
    downloading stream...
    >>> learner = MyBatchRegressor(stream.schema, batch_size=2)
    >>> for _ in range(4):
    ...     learner.train(stream.next_instance())
    [[0.49 0.07 0.   0.83 0.76 0.6  0.13 0.89 0.07 0.34]
     [0.22 0.4  0.66 0.53 0.84 0.71 0.58 0.47 0.57 0.53]]
    [17.95 13.81]
    <BLANKLINE>
    [[0.9  0.91 0.94 0.98 0.56 0.74 0.63 0.82 0.31 0.51]
     [0.79 0.86 0.36 0.84 0.16 0.95 0.11 0.29 0.41 0.99]]
    [20.77 18.3 ]
    <BLANKLINE>

    """

    def __init__(self, schema: Schema, batch_size: int, random_seed: int = 1) -> None:
        """Initialize the batch classifier.

        :param schema: A schema used to allocate memory for the batch.
        :param batch_size: The size of the batch.
        :param random_seed: The random seed for reproducibility.
        """
        super().__init__(schema, random_seed)
        self._batch = _BatchBuilder(
            batch_size, schema.get_num_attributes(), 1, np.float32, np.float32
        )

    def train(self, instance: RegressionInstance) -> None:
        """Collate instances into a batch and call :func:`batch_train`."""
        if self._batch.add(instance.x, instance.y_value):
            self.batch_train(self._batch.batch_x, self._batch.batch_y.flatten())

    @abstractmethod
    def batch_train(self, x: NDArray[np.number], y: NDArray[np.integer]) -> None:
        """Train the classifier with a batch of instances.

        :param x: A real valued matrix of shape ``(batch_size, num_attributes)``
            containing a batch of feature vectors.
        :param y: A real valued vector of shape ``(batch_size,)`` containing a batch
            of target values.
        """


class MOARegressor(Regressor):
    def __init__(self, schema=None, CLI=None, random_seed=1, moa_learner=None):
        super().__init__(schema=schema, random_seed=random_seed)
        self.CLI = CLI
        self.moa_learner = moa_learner

        if random_seed is not None:
            self.moa_learner.setRandomSeed(random_seed)

        if self.schema is not None:
            self.moa_learner.setModelContext(self.schema.get_moa_header())

        if self.CLI is not None:
            self.moa_learner.getOptions().setViaCLIString(CLI)

        self.moa_learner.prepareForUse()
        self.moa_learner.resetLearning()
        self.moa_learner.setModelContext(self.schema.get_moa_header())

    def __str__(self):
        full_name = str(self.moa_learner.getClass().getCanonicalName())
        return full_name.rsplit(".", 1)[1] if "." in full_name else full_name

    # def describe(self):
    #     return str(self.moa_learner)

    def CLI_help(self):
        return self.moa_learner.getOptions().getHelpString()

    def train(self, instance):
        self.moa_learner.trainOnInstance(instance.java_instance)

    def predict(self, instance):
        prediction_array = self.moa_learner.getVotesForInstance(instance.java_instance)
        # The learner didn't provide a prediction, returns 0.0 (probably the learner has not been initialised.)
        if len(prediction_array) == 0:
            return 0.0
        return prediction_array[0]


class SKRegressor(Regressor):
    """A wrapper class for using scikit-learn regressors in CapyMOA.

    Some of scikit-learn's regressors that are compatible with online learning
    have been wrapped and tested already in CapyMOA (See :mod:`capymoa.regressor`).

    However, if you want to use a scikit-learn regressor that has not been
    wrapped yet, you can use this class to wrap it yourself. This requires
    that the scikit-learn regressor implements the ``partial_fit`` and
    ``predict`` methods.

    For example, the following code demonstrates how to use a scikit-learn
    regressor in CapyMOA:

    >>> from sklearn.linear_model import SGDRegressor
    >>> from capymoa.datasets import Fried
    >>> stream = Fried()
    >>> sklearner = SGDRegressor(random_state=1)
    >>> learner = SKRegressor(sklearner, stream.schema)
    >>> for _ in range(10):
    ...     instance = stream.next_instance()
    ...     prediction = learner.predict(instance)
    ...     if prediction is not None:
    ...         print(f"y_value: {instance.y_value}, y_prediction: {prediction:.2f}")
    ...     else:
    ...         print(f"y_value: {instance.y_value}, y_prediction: None")
    ...     learner.train(instance)
    y_value: 17.949, y_prediction: None
    y_value: 13.815, y_prediction: 0.60
    y_value: 20.766, y_prediction: 1.30
    y_value: 18.301, y_prediction: 1.86
    y_value: 22.989, y_prediction: 2.28
    y_value: 25.986, y_prediction: 2.65
    y_value: 17.15, y_prediction: 3.51
    y_value: 14.006, y_prediction: 3.25
    y_value: 18.566, y_prediction: 3.80
    y_value: 12.107, y_prediction: 3.87

    A word of caution: even compatible scikit-learn regressors are not
    necessarily designed for online learning and might require some tweaking
    to work well in an online setting.

    See also :class:`capymoa.base.SKClassifier` for scikit-learn classifiers.
    """

    sklearner: _SKRegressorMixin
    """The underlying scikit-learn object."""

    def __init__(
        self, sklearner: _SKRegressorMixin, schema: Schema = None, random_seed: int = 1
    ):
        """Construct a scikit-learn regressor wrapper.

        :param sklearner: A scikit-learn classifier object to wrap that must
            implements ``partial_fit`` and ``predict``.
        :param schema: Describes the structure of the datastream.
        :param random_seed: Random seed for reproducibility.
        :raises ValueError: If the scikit-learn algorithm does not implement
            ``partial_fit`` or ``predict``.
        """
        super().__init__(schema=schema, random_seed=random_seed)

        # Checks if it implements partial_fit and predict
        if not hasattr(sklearner, "partial_fit") or not hasattr(sklearner, "predict"):
            raise ValueError(
                "Invalid scikit-learn algorithm provided. The algorithm does not implement partial_fit or predict. "
            )

        self.sklearner = sklearner
        self._trained_at_least_once = False

    def __str__(self):
        return str(self.sklearner)

    def train(self, instance: RegressionInstance):
        self.sklearner.partial_fit(
            [instance.x],
            [instance.y_value],
        )
        self._trained_at_least_once = True

    def predict(self, instance: Instance) -> float:
        if not self._trained_at_least_once:
            # scikit-learn does not allows invoking predict in a model that was not fit before
            return None
        return self.sklearner.predict([instance.x])[0]
