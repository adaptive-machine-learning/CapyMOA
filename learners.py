# Create the JVM and add the MOA jar to the classpath
from prepare_jpype import start_jpype
start_jpype()

from abc import ABC, abstractmethod

# MOA/Java imports
from moa.core import Utils

class Classifier(ABC):
    def __init__(self, schema=None, random_seed=1):
        self.random_seed = random_seed
        self.schema = schema

    @abstractmethod
    def __str__(self):
        pass

    @abstractmethod
    def train(self, instance):
        pass

    @abstractmethod
    def predict(self, instance):
        pass

    @abstractmethod
    def predict_proba(self, instance):
        pass

class MOAClassifier(Classifier):
    def __init__(self, schema=None, CLI=None, random_seed=1, moa_learner=None):
        super().__init__(schema=schema, random_seed=random_seed)
        self.CLI = CLI
        self.moa_learner = moa_learner

        self.moa_learner.setRandomSeed(self.random_seed)

        if self.schema is not None:
            self.moa_learner.setModelContext(self.schema.get_moa_header())

        if self.CLI is not None:
            self.moa_learner.getOptions().setViaCLIString(CLI)

        self.moa_learner.prepareForUse()
        self.moa_learner.resetLearningImpl()

    def __str__(self):
        # Remove the package information from the name of the learner. 
        full_name = str(self.moa_learner.getClass().getCanonicalName())
        return full_name.rsplit(".", 1)[1] if "." in full_name else full_name

    # def describe(self):
    #     return str(self.moa_learner)

    def CLI_help(self):
        return str(self.moa_learner.getOptions().getHelpString())

    def train(self, instance):
        self.moa_learner.trainOnInstance(instance.get_MOA_InstanceExample())

    def predict(self, instance):
        return Utils.maxIndex(self.moa_learner.getVotesForInstance(instance.get_MOA_InstanceExample()))

    def predict_proba(self, instance):
        return self.moa_learner.getVotesForInstance(instance.get_MOA_InstanceExample())


class SKClassifier(Classifier):
    def __init__(self, schema=None, random_seed=1, sklearner=None):
        super().__init__(schema=schema, random_seed=random_seed)
        self.sklearner = sklearner
        self.trained_at_least_once = False

    def __str__(self):
        return 'sklearner' # TODO: get the string describing the sklearner

    def train(self, instance):
        self.sklearner.partial_fit([instance.x()], [instance.y()], classes=self.schema.get_label_indexes())
        self.trained_at_least_once = True # deve (e tem que) ter um jeito melhor

    def predict(self, instance):
        if self.trained_at_least_once: # scikit-learn does not allows invoking predict in a model that was not fit before
            return self.sklearner.predict([instance.x()])
        else:
            return None

    def predict_proba(self, instance):
        if self.trained_at_least_once: # scikit-learn does not allows invoking predict in a model that was not fit before
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
        self.moa_learner.trainOnUnlabeledInstance(instance.get_MOA_InstanceExample().getData())

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
        prediction_array = self.moa_learner.getVotesForInstance(instance.get_MOA_InstanceExample())
        # The learner didn't provide a prediction, returns 0.0 (probably the learner has not been initialised.)
        if len(prediction_array) == 0:
            return 0.0
        return prediction_array[0]


