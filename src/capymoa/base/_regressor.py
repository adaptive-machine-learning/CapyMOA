from abc import ABC, abstractmethod

from sklearn.base import RegressorMixin as _SKRegressorMixin

from capymoa.instance import Instance, RegressionInstance
from capymoa.stream._stream import Schema
from capymoa.type_alias import TargetValue


class Regressor(ABC):
    def __init__(self, schema=None, random_seed=1):
        self.random_seed = random_seed
        self.schema = schema

    def __str__(self) -> str:
        return str(self.__class__.__name__)

    @abstractmethod
    def train(self, instance: RegressionInstance):
        pass

    @abstractmethod
    def predict(self, instance: RegressionInstance) -> TargetValue:
        pass


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

    def cli_help(self):
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
    >>> learner = SKRegressor(sklearner, stream.get_schema())
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
