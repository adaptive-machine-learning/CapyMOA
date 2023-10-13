# Create the JVM and add the MOA jar to the classpath
from prepare_jpype import start_jpype
start_jpype()

# import pandas as pd

# MOA/Java imports
from moa.core import Utils

class MOAClassifier:
    def __init__(self, schema=None, CLI=None, random_seed=1, moa_learner=None):
        self.schema = schema
        self.CLI = CLI
        self.random_seed = random_seed
        self.moa_learner = moa_learner

        self.moa_learner.setRandomSeed(random_seed)

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


class MOAClassifierSSL(MOAClassifier):
    def train_on_unlabeled(self, instance):
        self.moa_learner.trainOnUnlabeledInstance(instance.get_MOA_InstanceExample().getData())


class MOARegressor:
    def __init__(self, schema=None, CLI=None, random_seed=1, moa_learner=None):
        self.schema = schema
        self.CLI = CLI
        self.random_seed = random_seed
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
