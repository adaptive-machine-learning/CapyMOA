"""Let MOA ensembles use Python learners and drift detectors.

MOA builds ensemble members from a CLI string, which a Python object has none of.
But a MOA class option can hold a ready object. Here, JPype proxies implement the
MOA Java interfaces and call the Python object. The proxy is set as the option
value (see :func:`resolve_base_learner`).

Limits:

* MOA calls ``copy()`` to make each ensemble member. A proxy copy is a new
  untrained learner with the same parameters, not a clone of the trained state.
* Python code is called from Java, so it is slower than a MOA learner and is
  not safe for the multi-threaded (``number_of_jobs > 1``) ensembles.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from jpype import JArray, JDouble, JProxy
from moa.core import InstanceExample

from capymoa.base import Classifier, MOAClassifier, MOARegressor, Regressor
from capymoa.core import Instance, LabeledInstance, RegressionInstance
from capymoa.core.moa._cli import cli_str_classifier, cli_str_regressor
from capymoa.drift.base_detector import BaseDriftDetector, MOADriftDetector
from capymoa.stream import Schema


def _as_example(instance):
    """Wrap a MOA ``Instance`` as an ``InstanceExample`` (if it is not one)."""
    if isinstance(instance, InstanceExample):
        return instance
    return InstanceExample(instance)


def _java(obj):
    """Return a Java proxy of ``obj`` for the interfaces in ``obj._interfaces``.

    JPype calls the methods of ``obj`` by name when Java calls the proxy.
    """
    return JProxy(obj._interfaces, inst=obj)


class _LearnerProxy:
    """Shared code of the learner proxies. Use :func:`_java` to get the proxy."""

    _interfaces: tuple[str, ...]

    def __init__(self, learner: Classifier | Regressor):
        self._template = learner
        self._schema: Schema = learner.schema
        self._seed: int = learner.random_seed
        self._learner: Classifier | Regressor | None = None
        self._header = None
        self._seen = 0

    def _get(self) -> Classifier | Regressor:
        if self._learner is None:
            t = self._template
            self._learner = type(t).from_params(
                self._schema, t.get_params(), self._seed
            )
        return self._learner

    # Methods with logic.

    def copy(self):
        proxy = type(self)(self._template)
        proxy._schema, proxy._seed, proxy._header = (
            self._schema,
            self._seed,
            self._header,
        )
        return _java(proxy)

    def setModelContext(self, header):
        self._header = header
        self._schema = Schema(header)
        self._learner = None

    def getModelContext(self):
        return self._header

    def setRandomSeed(self, seed):
        self._seed = int(seed)
        self._learner = None

    def resetLearning(self):
        self._learner = None
        self._seen = 0

    def trainOnInstance(self, instance):
        self._train(_as_example(instance))
        self._seen += 1

    def getVotesForInstance(self, instance):
        return JArray(JDouble)(self._votes(_as_example(instance)))

    def trainingHasStarted(self):
        return self._seen > 0

    def trainingWeightSeenByModel(self):
        return float(self._seen)

    def isRandomizable(self):
        return True

    # Interface stubs.

    def correctlyClassifies(self, instance):
        votes = self.getVotesForInstance(instance)
        return len(votes) > 0 and int(np.argmax(list(votes))) == int(
            instance.classValue()
        )

    def getPredictionForInstance(self, instance):
        return None

    def getConfidenceForPrediction(self, instance, prediction):
        return 0.0

    def getSubClassifiers(self):
        return None

    def getSublearners(self):
        return None

    def getModel(self):
        return None

    def getModelMeasurements(self):
        return None

    def getOptions(self):
        return None

    def getPurposeString(self):
        return str(self._template)

    def getCLICreationString(self, requiredType):
        return str(self._template)

    def getAWTRenderer(self):
        return None

    def prepareForUse(self, *args):
        pass

    def getDescription(self, builder, indent):
        builder.append(str(self._template))

    def measureByteSize(self):
        return 0


class PythonClassifierProxy(_LearnerProxy):
    """A MOA ``Classifier`` that forwards to a CapyMOA :class:`Classifier`."""

    _interfaces = ("moa.classifiers.Classifier",)

    def _train(self, example):
        self._get().train(LabeledInstance(self._schema, example))

    def _votes(self, example) -> list[float]:
        proba = self._get().predict_proba(Instance(self._schema, example))
        # MOA reads an empty array as "no prediction".
        return [] if proba is None else np.asarray(proba, dtype=float).tolist()


class PythonRegressorProxy(_LearnerProxy):
    """A MOA regressor that forwards to a CapyMOA :class:`Regressor`.

    MOA regressors are ``Classifier`` objects that also implement the
    ``Regressor`` marker. They return the predicted value as the only vote.
    """

    _interfaces = ("moa.classifiers.Classifier", "moa.classifiers.Regressor")

    def _train(self, example):
        self._get().train(RegressionInstance(self._schema, example))

    def _votes(self, example) -> list[float]:
        value = self._get().predict(Instance(self._schema, example))
        return [] if value is None else [float(value)]


class PythonDriftDetectorProxy:
    """A MOA ``ChangeDetector`` that forwards to a CapyMOA drift detector."""

    _interfaces = ("moa.classifiers.core.driftdetection.ChangeDetector",)

    def __init__(self, detector: BaseDriftDetector):
        self._template = detector
        self._detector: BaseDriftDetector | None = None
        self._estimation = 0.0
        self._n = 0

    def _get(self) -> BaseDriftDetector:
        if self._detector is None:
            t = self._template
            self._detector = type(t).from_params(None, t.get_params())
        return self._detector

    def copy(self):
        return _java(PythonDriftDetectorProxy(self._template))

    def resetLearning(self):
        self._detector = None
        self._estimation, self._n = 0.0, 0

    def input(self, value):
        self._get().add_element(float(value))
        self._n += 1
        self._estimation += (float(value) - self._estimation) / self._n

    def getChange(self):
        return bool(self._get().detected_change())

    def getWarningZone(self):
        return bool(self._get().detected_warning())

    def getEstimation(self):
        return self._estimation

    def getDelay(self):
        return 0.0

    def getOutput(self):
        return JArray(JDouble)([])

    def getOptions(self):
        return None

    def getPurposeString(self):
        return str(self._template)

    def getCLICreationString(self, requiredType):
        return str(self._template)

    def prepareForUse(self, *args):
        pass

    def getDescription(self, builder, indent):
        builder.append(str(self._template))

    def measureByteSize(self):
        return 0


def resolve_base_learner(
    learner: Any, option: str = "baseLearner"
) -> tuple[str | None, dict[str, Any]]:
    """Turn a base learner into a CLI string and Java option values.

    MOA learners give their CLI string and no Java options. A Python learner gives
    ``None`` (leave it out of the CLI string) and a proxy to set on ``option``
    before MOA prepares the ensemble (see ``java_options`` of :class:`~capymoa.base.MOAClassifier`).

    >>> from capymoa.classifier import HoeffdingTree, PassiveAggressiveClassifier
    >>> from capymoa.datasets import ElectricityTiny
    >>> schema = ElectricityTiny().get_schema()
    >>> cli, options = resolve_base_learner(HoeffdingTree(schema))
    >>> cli.startswith('(trees.HoeffdingTree'), options
    (True, {})
    >>> cli, options = resolve_base_learner(PassiveAggressiveClassifier(schema))
    >>> cli, sorted(options)
    (None, ['baseLearner'])

    :param learner: A MOA or Python classifier.
    :param option: Name of the MOA class option that holds the base learner.
    :return: A CLI string (``None`` for a Python object) and a mapping from option
        name to Java object.
    :raises TypeError: If ``learner`` is not a classifier.
    """
    if isinstance(learner, Classifier) and not isinstance(learner, MOAClassifier):
        return None, {option: _java(PythonClassifierProxy(learner))}
    try:
        return cli_str_classifier(learner), {}
    except TypeError:
        raise TypeError(
            "base_learner must be a CapyMOA Classifier or a MOA classifier, "
            f"got {type(learner).__name__}."
        ) from None


def resolve_drift_detector(
    detector: Any, option: str
) -> tuple[str | None, dict[str, Any]]:
    """Like :func:`resolve_base_learner` for a drift detector.

    A MOA detector gives its CLI string (``None`` for a Python one, as the CLI
    keeps its default) and the Java option values.
    """
    from capymoa.core.moa._cli import cli_str_drift_detector

    if isinstance(detector, BaseDriftDetector) and not isinstance(
        detector, MOADriftDetector
    ):
        return None, {option: _java(PythonDriftDetectorProxy(detector))}
    try:
        return cli_str_drift_detector(detector), {}
    except TypeError:
        raise TypeError(
            "Drift detector must be a CapyMOA drift detector or a MOA detector, "
            f"got {type(detector).__name__}."
        ) from None


def resolve_base_regressor(
    learner: Any, option: str = "baseLearner"
) -> tuple[str | None, dict[str, Any]]:
    """Like :func:`resolve_base_learner` for a regressor.

    :param learner: A MOA or Python regressor.
    :param option: Name of the MOA class option that holds the learner.
    :return: A CLI string (``None`` for a Python object) and a mapping from option
        name to Java object.
    :raises TypeError: If ``learner`` is not a regressor.
    """
    if isinstance(learner, Regressor) and not isinstance(learner, MOARegressor):
        return None, {option: _java(PythonRegressorProxy(learner))}
    try:
        return cli_str_regressor(learner), {}
    except TypeError:
        raise TypeError(
            "base_learner must be a CapyMOA Regressor or a MOA regressor, "
            f"got {type(learner).__name__}."
        ) from None
