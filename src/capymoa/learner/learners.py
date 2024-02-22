from jpype import _jpype
from abc import ABC, abstractmethod, ABCMeta
from capymoa.stream.stream import Instance, Schema

# MOA/Java imports
from moa.core import Utils
from moa.classifiers import Classifier as MOA_Classifier_Interface


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
        f"{moa_learner_class_id_parts[-2]}.{moa_learner_class_id_parts[-1]}"
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

    # Check if the base_learner is a MOAClassifier
    if isinstance(learner, MOAClassifier):
        learner = _get_moa_creation_CLI(learner.moa_learner)

    # ... or a Classifier (Interface from MOA) type
    if isinstance(learner, MOA_Classifier_Interface):
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
    def train(self, instance: Instance):
        pass

    @abstractmethod
    def predict(self, instance: Instance):
        pass

    @abstractmethod
    def predict_proba(self, instance: Instance):
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

    def train(self, instance: Instance):
        self.moa_learner.trainOnInstance(instance.get_MOA_InstanceExample())

    def predict(self, instance: Instance):
        return self.schema.get_value_for_index(
            Utils.maxIndex(
                self.moa_learner.getVotesForInstance(instance.get_MOA_InstanceExample())
            )
        )

    def predict_proba(self, instance: Instance):
        return self.moa_learner.getVotesForInstance(instance.get_MOA_InstanceExample())


class SKClassifier(Classifier):
    """
    A wrapper class for using scikit-learn classifiers in CapyMOA

    Attributes:
    - schema (optional): The schema for the input instances. Defaults to None.
    - random_seed (optional): The random seed for reproducibility. Defaults to 1.
    - sklearner: The scikit-learn classifier object or class identifier.
    """

    def __init__(self, sklearner, schema=None, random_seed=1):
        super().__init__(schema=schema, random_seed=random_seed)

        # If sklearner is a class identifier instead of an object.
        if isinstance(sklearner, type):
            sklearner = sklearner()
        # Checks if it implements partial_fit and predict
        if not hasattr(sklearner, "partial_fit") or not hasattr(sklearner, "predict"):
            raise ValueError(
                "Invalid scikit-learn algorithm provided. The algorithm does not implement partial_fit or predict. "
            )

        self.sklearner = sklearner
        self.trained_at_least_once = False

    def __str__(self):
        return "sklearner"  # TODO: get the string describing the sklearner

    def train(self, instance):
        self.sklearner.partial_fit(
            [instance.x()], [instance.y_index()], classes=self.schema.get_label_indexes()
        )
        self.trained_at_least_once = True  # deve (e tem que) ter um jeito melhor

    def predict(self, instance):
        if (
            self.trained_at_least_once
        ):  # scikit-learn does not allows invoking predict in a model that was not fit before
            return self.sklearner.predict([instance.x()])[0]
        else:
            return None

    def predict_proba(self, instance):
        if (
            self.trained_at_least_once
        ):  # scikit-learn does not allows invoking predict in a model that was not fit before
            return self.sklearner.predict_proba([instance.x()])
        else:
            return None


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
        self.moa_learner.trainOnUnlabeledInstance(
            instance.get_MOA_InstanceExample().getData()
        )


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
    def train(self, instance):
        pass

    @abstractmethod
    def predict(self, instance):
        pass


class MOARegressor(Regressor):
    def __init__(self, schema=None, CLI=None, random_seed=1, moa_learner=None):
        super().__init__(schema=schema, random_seed=random_seed)
        self.CLI = CLI
        self.moa_learner = moa_learner

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
        self.moa_learner.trainOnInstance(instance.get_MOA_InstanceExample())

    def predict(self, instance):
        prediction_array = self.moa_learner.getVotesForInstance(
            instance.get_MOA_InstanceExample()
        )
        # The learner didn't provide a prediction, returns 0.0 (probably the learner has not been initialised.)
        if len(prediction_array) == 0:
            return 0.0
        return prediction_array[0]
