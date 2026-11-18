from abc import ABC, abstractmethod
from collections.abc import Mapping
from typing import Any, overload

import numpy as np
from jpype import _jpype
from sklearn.base import ClassifierMixin as _SKClassifierMixin

from capymoa.base._learner_params import LearnerParamsMixin
from capymoa.core import Instance, LabeledInstance, LabelIndex, LabelProbabilities
from capymoa.core.moa._options import set_moa_options
from capymoa.stream._stream import Schema


class Classifier(LearnerParamsMixin, ABC):
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
    def predict_proba(self, instance: Instance) -> LabelProbabilities:
        """Return probability estimates for each label.

        Must work at any time, even before training. If the classifier can not
        predict yet, return a uniform distribution over the labels.

        :param instance: The instance to estimate the probabilities for.
        :return: An array of probabilities for each label.
        """

    def predict(self, instance: Instance) -> LabelIndex:
        """Predict the label of an instance.

        The base implementation calls :func:`predict_proba` and returns the
        label with the highest probability.

        :param instance: The instance to predict the label for.
        :return: The predicted label.
        """
        return self._argmax(self.predict_proba(instance))

    def _argmax(self, proba: LabelProbabilities) -> LabelIndex:
        """Index of the largest probability, breaking exact ties at random.

        A learner that can not predict yet returns a uniform distribution. A
        plain ``argmax`` would always pick label 0, so we pick among the tied
        labels.

        The generator is a Mersenne Twister with legacy seeding, as in the SSL
        evaluation loop. MOA's Java loop uses the same generator (commons-math
        ``MersenneTwister(seed)``). Both draw one raw 32-bit output ``r`` per
        tie, and pick tied label ``(r * k) >> 32`` of the ``k`` tied labels.
        """
        proba = np.asarray(proba).ravel()
        tied = np.flatnonzero(proba == proba.max())
        if len(tied) == 1:
            return int(tied[0])
        if getattr(self, "_tie_mt", None) is None:
            self._tie_mt = np.random.MT19937()
            self._tie_mt._legacy_seeding(getattr(self, "random_seed", 1))
        raw = int(self._tie_mt.random_raw())
        return int(tied[(raw * len(tied)) >> 32])

    def _uniform_proba(self) -> LabelProbabilities:
        """Uniform distribution over the labels in the schema."""
        n = self.schema.get_num_classes()
        return np.full(n, 1.0 / n, dtype=np.float64)


class MOAClassifier(Classifier):
    """
    A wrapper class for using MOA (Massive Online Analysis) classifiers in CapyMOA.

    Attributes:
    - schema: The schema representing the instances. Defaults to None.
    - CLI: The command-line interface (CLI) configuration for the MOA learner.
    - random_seed: The random seed for reproducibility. Defaults to 1.
    - moa_learner: The MOA learner object or class identifier.
    """

    @overload
    def __init__(
        self,
        moa_learner,
        schema=None,
        CLI: str | None = None,
        random_seed=1,
    ): ...

    @overload
    def __init__(
        self,
        moa_learner,
        schema=None,
        CLI: None = None,
        random_seed=1,
        *,
        options: Mapping[str, Any],
    ): ...

    def __init__(
        self,
        moa_learner,
        schema=None,
        CLI=None,
        random_seed=1,
        options: Mapping[str, Any] | None = None,
    ):
        """
        :param options: MOA option names mapped to values. Use instead of CLI for values
            with no CLI string, such as Java objects.
        """
        super().__init__(schema=schema, random_seed=random_seed)
        self.CLI = CLI
        # If moa_learner is a class identifier instead of an object
        if isinstance(moa_learner, type):
            if isinstance(moa_learner, _jpype._JClass):
                moa_learner = moa_learner()
            else:  # this is not a Java object, thus it certainly isn't a MOA learner
                raise TypeError("Invalid MOA classifier provided.")
        self.moa_learner = moa_learner

        self.moa_learner.setRandomSeed(self.random_seed)

        if self.schema is not None:
            self.moa_learner.setModelContext(self.schema.get_moa_header())

        set_moa_options(self.moa_learner, self.CLI, options)

        self.moa_learner.prepareForUse()
        self.moa_learner.resetLearningImpl()
        self.moa_learner.setModelContext(schema.get_moa_header())

    def __str__(self):
        # Removes the package information from the name of the learner.
        full_name = str(self.moa_learner.getClass().getCanonicalName())
        return full_name.rsplit(".", 1)[1] if "." in full_name else full_name

    def cli_help(self):
        return str(self.moa_learner.getOptions().getHelpString())

    def train(self, instance):
        self.moa_learner.trainOnInstance(instance.java_instance)

    def predict(self, instance) -> LabelIndex:
        # Same as MOA: the first label with the highest vote, or label 0 when the
        # learner gives no votes (MOA's ``Utils.maxIndex``). No random tie-break.
        votes = self.moa_learner.getVotesForInstance(instance.java_instance)
        best, best_index = None, 0
        for i, vote in enumerate(votes):
            # ``nan > x`` is false, so NaN never wins (unlike ``np.argmax``).
            if best is None or vote > best:
                best, best_index = vote, i
        return best_index

    def predict_proba(self, instance) -> LabelProbabilities:
        votes = np.array(self.moa_learner.getVotesForInstance(instance.java_instance))
        total = votes.sum()
        # MOA returns unnormalised scores. Their scale depends on the learner
        # (counts for majority class, tiny likelihood products for Naive Bayes).
        # Votes that carry no prediction (empty, non-finite or all zero) give a
        # uniform distribution.
        if votes.size == 0 or not np.isfinite(total) or total <= 0.0:
            return self._uniform_proba()
        # MOA sizes votes by the classes seen so far. Pad with zeros to match the schema.
        num_classes = self.schema.get_num_classes()
        if votes.size < num_classes:
            votes = np.pad(votes, (0, num_classes - votes.size))
        return votes / total


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
    >>> learner = SKClassifier(sklearner, stream.get_schema())
    >>> for _ in range(10):
    ...     instance = stream.next_instance()
    ...     prediction = learner.predict(instance)
    ...     print(f"True: {instance.y_index}, Predicted: {prediction}")
    ...     learner.train(instance)
    True: 1, Predicted: 0
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

    def predict(self, instance: Instance) -> LabelIndex:
        if not self._trained_at_least_once:
            # scikit-learn does not allows invoking predict in a model that was not fit before
            return self._argmax(self._uniform_proba())
        return int(self.sklearner.predict([instance.x])[0])

    def predict_proba(self, instance: Instance) -> LabelProbabilities:
        if not self._trained_at_least_once:
            # scikit-learn does not allows invoking predict in a model that was not fit before
            return self._uniform_proba()
        elif not hasattr(self.sklearner, "predict_proba"):
            # When classifiers do not implement predict_proba use a one-hot encoding
            proba = np.zeros((self.schema.get_num_classes(),), dtype=np.float64)
            proba[self.predict(instance)] = 1.0
            return proba
        else:
            # Default case passing through predict_proba
            return self.sklearner.predict_proba([instance.x])[0]
