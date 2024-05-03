from abc import ABC, abstractmethod
from typing import Optional, Union

from jpype import _jpype
from moa.classifiers import (
    Classifier as MOA_Classifier_Interface,
    Regressor as MOA_Regressor_Interface,
)
from moa.classifiers.predictioninterval import PredictionIntervalLearner as MOA_PredictionInterval_Interface
from moa.classifiers.trees import (
ARFFIMTDD as MOA_ARFFIMTDD,
SelfOptimisingBaseTree as MOA_SOKNLBT,
)
from moa.core import Utils

from capymoa.instance import Instance, LabeledInstance, RegressionInstance
from capymoa.stream._stream import Schema
from capymoa.type_alias import LabelIndex, LabelProbabilities, TargetValue

from sklearn.base import ClassifierMixin as _SKClassifierMixin
from sklearn.base import RegressorMixin as _SKRegressorMixin

##############################################################
##################### INTERNAL FUNCTIONS #####################
##############################################################


def _get_moa_creation_CLI(moa_learner):
    """
    Auxiliary function to retrieve the command-line interface (CLI)
    creation command for a MOA learner.

    Parameters:
    - moa_learner: The MOA learner for which the CLI command is needed,
    it must be a MOA learner object

    Returns:
    A string representing the CLI command for creating the MOA learner.
    """
    moa_learner_class_id = str(moa_learner.getClass().getName())
    moa_learner_class_id_parts = moa_learner_class_id.split(".")

    moa_learner_str = (
        f"{moa_learner_class_id_parts[-1]}" if isinstance(moa_learner, MOA_ARFFIMTDD) or isinstance(moa_learner, MOA_SOKNLBT)
        else f"{moa_learner_class_id_parts[-2]}.{moa_learner_class_id_parts[-1]}"
    )

    moa_cli_creation = str(moa_learner.getCLICreationString(moa_learner.__class__))
    CLI = moa_cli_creation.split(" ", 1)

    if len(CLI) > 1 and len(CLI[1]) > 1:
        moa_learner_str = f"({moa_learner_str} {CLI[1]})"

    return moa_learner_str


def _extract_moa_learner_CLI(learner):
    """
    Auxiliary function to extract the command-line interface (CLI)
    from a MOA learner object or a MOA learner class or even a
    MOAClassifier object (which has a moa_learner internally).

    Parameters:
    - moa_learner: The object or class representing the MOA learner

    Returns:
    A string representing the CLI command for creating the MOA learner.
    """

    # Check if the base_learner is a MOAClassifie or a MOARegressor
    if isinstance(learner, MOAClassifier) or isinstance(learner, MOARegressor) or isinstance(learner, MOAPredictionIntervalLearner):
        learner = _get_moa_creation_CLI(learner.moa_learner)

    # ... or a Classifier or a Regressor (Interfaces from MOA) type
    if isinstance(learner, MOA_Classifier_Interface) or isinstance(learner, MOA_Regressor_Interface) or isinstance(learner, MOA_PredictionInterval_Interface):
        learner = _get_moa_creation_CLI(learner)

    # ... or a java object, which we presume is a MOA object (if it is not, MOA will raise the error)
    if type(learner) == _jpype._JClass:
        learner = _get_moa_creation_CLI(learner())
    return learner


##############################################################
######################### CLASSIFIERS ########################
##############################################################


class Classifier(ABC):
    """
    Abstract base class for machine learning classifiers.

    Attributes:
    - schema: The schema representing the instances. Defaults to None.
    - random_seed: The random seed for reproducibility. Defaults to 1.
    """

    def __init__(self, schema: Schema, random_seed=1):
        self.random_seed = random_seed
        self.schema = schema
        if self.schema is None:
            raise ValueError("Schema must be initialised")

    @abstractmethod
    def __str__(self):
        pass

    @abstractmethod
    def train(self, instance: LabeledInstance):
        pass

    @abstractmethod
    def predict(self, instance: Instance) -> Optional[LabelIndex]:
        pass

    @abstractmethod
    def predict_proba(self, instance: Instance) -> LabelProbabilities:
        pass


class MOAClassifier(Classifier):
    """
    A wrapper class for using MOA (Massive Online Analysis) classifiers in CapyMOA.

    Attributes:
    - schema: The schema representing the instances. Defaults to None.
    - CLI: The command-line interface (CLI) configuration for the MOA learner.
    - random_seed: The random seed for reproducibility. Defaults to 1.
    - moa_learner: The MOA learner object or class identifier.
    """

    def __init__(self, moa_learner, schema=None, CLI=None, random_seed=1):
        super().__init__(schema=schema, random_seed=random_seed)
        self.CLI = CLI
        # If moa_learner is a class identifier instead of an object
        if isinstance(moa_learner, type):
            if type(moa_learner) == _jpype._JClass:
                moa_learner = moa_learner()
            else:  # this is not a Java object, thus it certainly isn't a MOA learner
                raise ValueError("Invalid MOA classifier provided.")
        self.moa_learner = moa_learner

        self.moa_learner.setRandomSeed(self.random_seed)

        if self.schema is not None:
            self.moa_learner.setModelContext(self.schema.get_moa_header())

        # If the CLI is None, we assume the object has already been configured
        # or that default values should be used.
        if self.CLI is not None:
            self.moa_learner.getOptions().setViaCLIString(CLI)

        self.moa_learner.prepareForUse()
        self.moa_learner.resetLearningImpl()

    def __str__(self):
        # Removes the package information from the name of the learner.
        full_name = str(self.moa_learner.getClass().getCanonicalName())
        return full_name.rsplit(".", 1)[1] if "." in full_name else full_name

    def CLI_help(self):
        return str(self.moa_learner.getOptions().getHelpString())

    def train(self, instance):
        self.moa_learner.trainOnInstance(instance.java_instance)

    def predict(self, instance):
        return Utils.maxIndex(
            self.moa_learner.getVotesForInstance(instance.java_instance)
        )

    def predict_proba(self, instance):
        return self.moa_learner.getVotesForInstance(instance.java_instance)


class SKClassifier(Classifier):
    """A wrapper class for using scikit-learn classifiers in CapyMOA.

    Some of scikit-learn's classifiers that are compatible with online learning
    have been wrapped and tested already in CapyMOA (See :mod:`capymoa.classifier`). 
    
    However, if you want to use a scikit-learn classifier that has not been
    wrapped yet, you can use this class to wrap it yourself. This requires
    that the scikit-learn classifier implements the ``partial_fit`` and
    ``predict`` methods.

    For example, the following code demonstrates how to use a scikit-learn
    classifier in CapyMOA:

    >>> from sklearn.linear_model import SGDClassifier
    >>> from capymoa.base import SKClassifier
    >>> from capymoa.datasets import ElectricityTiny
    >>> stream = ElectricityTiny()
    >>> sklearner = SGDClassifier(random_state=1)
    >>> learner = SKClassifier(sklearner, stream.schema)
    >>> for _ in range(10):
    ...     instance = stream.next_instance()
    ...     prediction = learner.predict(instance)
    ...     print(f"True: {instance.y_index}, Predicted: {prediction}")
    ...     learner.train(instance)
    True: 1, Predicted: None
    True: 1, Predicted: 1
    True: 1, Predicted: 1
    True: 1, Predicted: 1
    True: 0, Predicted: 1
    True: 0, Predicted: 1
    True: 0, Predicted: 0
    True: 0, Predicted: 0
    True: 0, Predicted: 0
    True: 0, Predicted: 0

    A word of caution: even compatible scikit-learn classifiers are not
    necessarily designed for online learning and might require some tweaking
    to work well in an online setting.

    See also :class:`capymoa.base.SKRegressor` for scikit-learn regressors.
    """

    sklearner: _SKClassifierMixin
    """The underlying scikit-learn object."""

    def __init__(self, sklearner: _SKClassifierMixin, schema: Schema = None, random_seed: int = 1):
        """Construct a scikit-learn classifier wrapper.

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

    def train(self, instance: LabeledInstance):
        self.sklearner.partial_fit(
            [instance.x],
            [instance.y_index],
            classes=self.schema.get_label_indexes(),
        )
        self._trained_at_least_once = True

    def predict(self, instance: Instance):
        if not self._trained_at_least_once:
            # scikit-learn does not allows invoking predict in a model that was not fit before
            return None
        return self.sklearner.predict([instance.x])[0]

    def predict_proba(self, instance: Instance):
        if not self._trained_at_least_once:
            # scikit-learn does not allows invoking predict in a model that was not fit before
            return None
        self.sklearner.predict_proba([instance.x])


##############################################################
############################# SSL ############################
##############################################################


class ClassifierSSL(Classifier):
    def __init__(self, schema=None, random_seed=1):
        super().__init__(schema=schema, random_seed=random_seed)

    @abstractmethod
    def train_on_unlabeled(self, instance):
        pass


# Multiple inheritance
class MOAClassifierSSL(MOAClassifier, ClassifierSSL):
    def train_on_unlabeled(self, instance):
        self.moa_learner.trainOnUnlabeledInstance(instance.java_instance.getData())


##############################################################
######################### REGRESSORS #########################
##############################################################


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

    def __init__(self, sklearner: _SKRegressorMixin, schema: Schema = None, random_seed: int = 1):
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


### Prediction Interval Learner ###
class PredictionIntervalLearner(Regressor):
    def __init__(self, schema=None, random_seed=1):
        super().__init__(schema=schema, random_seed=random_seed)

    @abstractmethod
    def train(self, instance):
        pass
    @abstractmethod
    def predict(self, instance):
        pass


class MOAPredictionIntervalLearner(MOARegressor, PredictionIntervalLearner):

    def train(self, instance):
        self.moa_learner.trainOnInstance(instance.java_instance)
    def predict(self, instance):
        prediction_PI = self.moa_learner.getVotesForInstance(instance.java_instance)
        if len(prediction_PI) != 3:
            return [0, 0, 0]
        else:
            return prediction_PI
