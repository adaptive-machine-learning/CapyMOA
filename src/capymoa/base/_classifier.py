from abc import ABC, abstractmethod
from typing import Optional

import numpy as np
from jpype import _jpype
from moa.core import Utils
from numpy.typing import NDArray
from sklearn.base import ClassifierMixin as _SKClassifierMixin

from capymoa.instance import Instance, LabeledInstance
from capymoa.stream._stream import Schema
from capymoa.type_alias import LabelIndex, LabelProbabilities


class Classifier(ABC):
    """Base class for classifiers.

    In machine learning, a classifier is a supervised learner that assigns a
    label to an instance. The label can be a class, a category, or other nominal
    value.
    """

    random_seed: int
    """The random seed for reproducibility.
    
    When implementing a classifier ensure random number generators are seeded.
    """

    schema: Schema
    """The schema representing the instances."""

    def __init__(self, schema: Schema, random_seed: int = 1):
        self.random_seed = random_seed
        self.schema = schema

    def __str__(self) -> str:
        """Return a label/name for the classifier.

        Used for labeling classifiers in visualizations.
        """
        return self.__class__.__name__

    @abstractmethod
    def train(self, instance: LabeledInstance) -> None:
        """Train the classifier with a labeled instance.

        :param instance: The labeled instance to train the classifier with.
        """

    @abstractmethod
    def predict_proba(self, instance: Instance) -> Optional[LabelProbabilities]:
        """Return probability estimates for each label.

        :param instance: The instance to estimate the probabilities for.
        :return: An array of probabilities for each label or ``None`` if the
            classifier is unable to make a prediction.
        """

    def predict(self, instance: Instance) -> Optional[LabelIndex]:
        """Predict the label of an instance.

        The base implementation calls :func:`predict_proba` and returns the
        label with the highest probability.

        :param instance: The instance to predict the label for.
        :return: The predicted label or ``None`` if the classifier is unable
            to make a prediction.
        """
        prediction = self.predict_proba(instance)
        return np.argmax(prediction) if prediction is not None else None


class BatchClassifier(Classifier):
    """Base class for classifiers that support mini-batches.

    Supported by:

    - :func:`capymoa.ocl.evaluation.ocl_train_eval_loop`
    - :func:`capymoa.evaluation.prequential_evaluation`

    Evaluators that support batch classifiers will call the :func:`batch_train`
    and :func:`batch_predict_proba` methods instead of :func:`train` and
    :func:`predict_proba`:

    >>> from capymoa.base import BatchClassifier
    >>> from capymoa.datasets import ElectricityTiny
    >>> from capymoa.evaluation import prequential_evaluation
    >>>
    >>> batch_size = 500
    >>> class MyBatchClassifier(BatchClassifier):
    ...     def batch_train(self, x, y):
    ...         print(f"batch_train x.shape={x.shape} x.dtype={x.dtype}")
    ...         print(f"batch_train y.shape={y.shape} y.dtype={y.dtype}")
    ...
    ...     def batch_predict_proba(self, x):
    ...         print(f"batch_predict_proba x.shape={x.shape} x.dtype={x.dtype}")
    ...         return np.zeros((x.shape[0], self.schema.get_num_classes()))
    ...
    >>> stream = ElectricityTiny()
    >>> batch_classifier = MyBatchClassifier(stream.schema)
    >>> _ = prequential_evaluation(
    ...     stream,
    ...     batch_classifier,
    ...     batch_size=batch_size,
    ...     max_instances=721
    ... )
    batch_predict_proba x.shape=(500, 6) x.dtype=float64
    batch_train x.shape=(500, 6) x.dtype=float64
    batch_train y.shape=(500,) y.dtype=int64
    batch_predict_proba x.shape=(221, 6) x.dtype=float64
    batch_train x.shape=(221, 6) x.dtype=float64
    batch_train y.shape=(221,) y.dtype=int64

    You can manually use ``itertools.batched`` (python 3.12) function and
    ``np.stack`` to collect batches of instances as a matrix:

    >>> from itertools import islice
    >>> from capymoa._utils import batched # Not available in python < 3.12
    >>> stream.restart() # streams are stateful, so restart it
    >>> for i, batch in enumerate(batched(stream, 100)):
    ...     x = np.stack([instance.x for instance in batch])
    ...     y = np.stack([instance.y_index for instance in batch])
    ...     batch_classifier.batch_train(x, y)
    ...     break
    batch_train x.shape=(100, 6) x.dtype=float64
    batch_train y.shape=(100,) y.dtype=int64

    The default implementation of :func:`train` and :func:`predict` calls the
    batch variants with a batch of size 1. This is useful for parts of CapyMOA
    that expect a classifier to be able to train and predict on single
    instances.

    >>> instance = next(stream)
    >>> batch_classifier.train(instance)
    batch_train x.shape=(1, 6) x.dtype=float64
    batch_train y.shape=(1,) y.dtype=int64
    >>> batch_classifier.predict(instance)
    batch_predict_proba x.shape=(1, 6) x.dtype=float64
    np.int64(0)
    >>> batch_classifier.predict_proba(instance)
    batch_predict_proba x.shape=(1, 6) x.dtype=float64
    array([0., 0.])
    """

    @abstractmethod
    def batch_train(self, x: NDArray[np.number], y: NDArray[np.integer]) -> None:
        """Train the classifier with a batch of instances.

        :param x: A real valued matrix of shape ``(batch_size, num_attributes)``
            containing a batch of feature vectors.
        :param y: An integer array of shape ``(batch_size,)`` containing the
            label index. Missing labels are coded as ``-1`` in the
            semi-supervised setting.
        """

    @abstractmethod
    def batch_predict_proba(self, x: NDArray[np.number]) -> NDArray[np.number]:
        """Return probability estimates for each label in a batch.

        :param x: A real valued matrix of shape ``(batch_size, num_attributes)``
            containing a batch of feature vectors.
        :return: An array of shape ``(batch_size, num_labels)`` containing the
            probabilities for each label.
        """

    def train(self, instance: LabeledInstance) -> None:
        """Calls :func:`batch_train` with a batch of size 1."""
        return self.batch_train(
            x=instance.x.reshape(1, -1), y=np.array(instance.y_index).reshape(-1)
        )

    def predict_proba(self, instance: Instance) -> Optional[LabelProbabilities]:
        """Calls :func:`batch_predict_proba` with a batch of size 1."""
        return self.batch_predict_proba(x=instance.x.reshape(1, -1))[0]

    def batch_predict(self, x: NDArray[np.number]) -> NDArray[np.integer]:
        """Predict the labels of a batch of instances.

        :param x: A real valued matrix of shape ``(batch_size, num_attributes)``
            containing a batch of feature vectors.
        :return: An array of shape ``(batch_size,)`` containing the predicted
            labels.
        """
        return np.argmax(self.batch_predict_proba(x), axis=1)


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
            if isinstance(moa_learner, _jpype._JClass):
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
        self.moa_learner.setModelContext(schema.get_moa_header())

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

    def __init__(
        self, sklearner: _SKClassifierMixin, schema: Schema = None, random_seed: int = 1
    ):
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
