from ._classifier import Classifier, MOAClassifier, BatchClassifier
from ._base import Instance
from abc import abstractmethod


class ClassifierSSL(Classifier):
    """Base class for semi-supervised learning classifiers."""

    @abstractmethod
    def train_on_unlabeled(self, instance: Instance):
        pass


class MOAClassifierSSL(MOAClassifier, ClassifierSSL):
    """Wrapper for using MOA semi-supervised learning classifiers."""

    def train_on_unlabeled(self, instance: Instance):
        self.moa_learner.trainOnUnlabeledInstance(instance.java_instance.getData())


class BatchClassifierSSL(BatchClassifier, ClassifierSSL):
    """Base class for semi-supervised learning batch classifiers."""

    def train_on_unlabeled(self, instance: Instance):
        if self._batch.add(instance.x, -1):
            self.batch_train(self._batch.batch_x, self._batch.batch_y.flatten())
