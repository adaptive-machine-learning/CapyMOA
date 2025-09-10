from abc import abstractmethod

from ._base import Instance
from ._classifier import Classifier, MOAClassifier


class ClassifierSSL(Classifier):
    """Base class for semi-supervised learning classifiers."""

    @abstractmethod
    def train_on_unlabeled(self, instance: Instance):
        pass


class MOAClassifierSSL(MOAClassifier, ClassifierSSL):
    """Wrapper for using MOA semi-supervised learning classifiers."""

    def train_on_unlabeled(self, instance: Instance):
        self.moa_learner.trainOnUnlabeledInstance(instance.java_instance.getData())
