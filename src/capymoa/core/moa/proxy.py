"""Adapters let moa use python components via jpype proxies.

Wraps a Python learner as a JPype proxy for use with MOA. Useful for ensembles and
other scenarios where a Python learner needs to be used within the MOA framework.
"""

from __future__ import annotations

from collections.abc import Collection, MutableSequence
from typing import Any, cast

import jpype
import numpy as np
from com.github.javacliparser import Options
from com.yahoo.labs.samoa.instances import (
    Instance as SamoaInstance,
)
from com.yahoo.labs.samoa.instances import (
    InstancesHeader,
    Prediction,
)
from moa import MOAObject
from moa.classifiers import Classifier as JavaClassifier
from moa.classifiers import OneClassClassifier as JavaOneClassClassifier
from moa.classifiers import Regressor as JavaRegressor
from moa.classifiers import SemiSupervisedLearner as JavaSemiSupervisedLearner
from moa.classifiers.core.driftdetection import ChangeDetector as JavaChangeDetector
from moa.classifiers.predictioninterval import (
    PredictionIntervalLearner as JavaPredictionIntervalLearner,
)
from moa.cluster import Clustering, SphereCluster
from moa.clusterers import Clusterer as JavaClusterer
from moa.core import Example, InstanceExample, Measurement, ObjectRepository
from moa.gui import AWTRenderer
from moa.learners import Learner
from moa.learners.featureanalysis import (
    FeatureImportanceClassifier as JavaFeatureImportanceClassifier,
)
from moa.tasks import TaskMonitor

from capymoa.base import (
    AnomalyDetector,
    Classifier,
    ClassifierSSL,
    Clusterer,
    ClusteringResult,
    LearnerSpec,
    PredictionIntervalLearner,
    Regressor,
    learner_from_params,
)
from capymoa.core import LabeledInstance, RegressionInstance
from capymoa.drift.base_detector import BaseDriftDetector
from capymoa.feature import FeatureImportanceClassifier
from capymoa.stream import Schema


class ProxyNotImplementedError(NotImplementedError):
    """Exception raised for methods that are not implemented in the proxy class."""

    def __init__(self, parent: Any, fn_name: str):
        super().__init__(
            f"{parent._learner_spec['learner']}: {fn_name} method is not implemented in the proxy class."
        )


def _inherit_overrides(cls: type) -> type:
    """Copy inherited ``@JOverride`` methods onto ``cls``.

    ``jpype.JImplements`` only checks methods defined on the class itself, so
    methods from a base class are not seen. Apply below ``jpype.JImplements``.
    """
    for base in cls.__mro__[1:]:
        for name, value in vars(base).items():
            if name not in cls.__dict__ and hasattr(value, "__joverride__"):
                setattr(cls, name, value)
    return cls


class _ProxyOptionHandler:
    """Shared ``OptionHandler`` stubs for all proxies. Subclasses must use
    :func:`_inherit_overrides`."""

    def __init__(self, learner_spec: LearnerSpec):
        self._learner_spec = learner_spec

    def __str__(self) -> str:
        return self._learner_spec["learner"].rsplit(".", 1)[-1]

    @jpype.JOverride
    def getPurposeString(self) -> str:
        raise ProxyNotImplementedError(self, "getPurposeString")

    @jpype.JOverride
    def getOptions(self) -> Options:
        raise ProxyNotImplementedError(self, "getOptions")

    @jpype.JOverride
    def getCLICreationString(self, expected_type: type[Any]) -> str:
        raise ProxyNotImplementedError(self, "getCLICreationString")

    @jpype.JOverride
    def measureByteSize(self) -> int:
        raise ProxyNotImplementedError(self, "measureByteSize")

    @jpype.JOverride
    def getDescription(self, sb: Any, indent: int) -> None:
        raise ProxyNotImplementedError(self, "getDescription")


class _ProxyLearner(_ProxyOptionHandler):
    """Shared learner code for classifier and clusterer proxies.

    Builds the python learner from ``learner_spec`` in :meth:`prepareForUse`.
    """

    def __init__(self, learner_spec: LearnerSpec):
        super().__init__(learner_spec)
        self._py_learner: Any = None
        # MOA learners also default to a seed of 1.
        self._random_seed: int = 1
        self._schema: Schema | None = None

    def _to_py(self, instance: SamoaInstance | Example[SamoaInstance]) -> Any:
        """Convert a Java instance to a python one, like a stream would."""
        assert self._schema is not None, "Schema is not set"
        if not isinstance(instance, InstanceExample):
            instance = InstanceExample(instance)
        if self._schema.is_regression():
            return RegressionInstance.from_java_instance(self._schema, instance)
        return LabeledInstance.from_java_instance(self._schema, instance)

    @jpype.JOverride
    def copy(self) -> Any:
        return type(self)(self._learner_spec)

    @jpype.JOverride
    def isRandomizable(self) -> bool:
        return True

    @jpype.JOverride
    def setRandomSeed(self, s: int) -> None:
        self._random_seed = s

    @jpype.JOverride
    def resetLearningImpl(self) -> None:
        return self.prepareForUse()

    @jpype.JOverride
    def resetLearning(self) -> None:
        return self.resetLearningImpl()

    @jpype.JOverride
    def setModelContext(self, ih: InstancesHeader) -> None:
        self._schema = Schema(ih)

    @jpype.JOverride
    def getModelContext(self) -> InstancesHeader:
        if self._schema is None:
            raise RuntimeError("Set `setModelContext` before calling `getModelContext`")
        return self._schema.get_moa_header()

    @jpype.JOverride
    def prepareForUse(
        self,
        monitor: TaskMonitor | None = None,
        repository: ObjectRepository | None = None,
    ) -> None:
        self._py_learner = learner_from_params(
            self._learner_spec, self._schema, self._random_seed
        )

    @jpype.JOverride
    def trainingHasStarted(self) -> bool:
        raise ProxyNotImplementedError(self, "trainingHasStarted")

    @jpype.JOverride
    def trainingWeightSeenByModel(self) -> float:
        raise ProxyNotImplementedError(self, "trainingWeightSeenByModel")

    @jpype.JOverride
    def getModelMeasurements(self) -> MutableSequence[Measurement]:
        raise ProxyNotImplementedError(self, "getModelMeasurements")

    @jpype.JOverride
    def getAWTRenderer(self) -> AWTRenderer:
        raise ProxyNotImplementedError(self, "getAWTRenderer")


@jpype.JImplements(JavaClassifier)
@_inherit_overrides
class ProxyJavaClassifier(_ProxyLearner):
    """An adapter implementing moa's ``moa.classifiers.Classifier`` interface in python.

    https://github.com/Waikato/moa/blob/master/moa/src/main/java/moa/classifiers/Classifier.java#L35
    """

    @jpype.JOverride
    def getSubClassifiers(self) -> MutableSequence[JavaClassifier]:
        raise ProxyNotImplementedError(self, "getSubClassifiers")

    @jpype.JOverride
    def correctlyClassifies(self, instance: SamoaInstance) -> bool:
        raise ProxyNotImplementedError(self, "correctlyClassifies")

    @property
    def _py_classifier(self) -> Classifier:
        assert self._py_learner is not None, "Python learner is not initialized"
        return self._py_learner

    @jpype.JOverride
    def trainOnInstance(
        self, instance_or_example: SamoaInstance | Example[SamoaInstance]
    ) -> None:
        self._py_classifier.train(self._to_py(instance_or_example))

    @jpype.JOverride
    def getVotesForInstance(
        self, instance_or_example: SamoaInstance | Example[SamoaInstance]
    ) -> MutableSequence[float]:
        assert self._schema is not None, "Schema is not set"
        py_instance = self._to_py(instance_or_example)
        y_hat = self._py_classifier.predict_proba(py_instance)

        if y_hat is None:
            n_classes = self._schema.get_num_classes()
            y_hat = np.full(n_classes, 1.0 / n_classes)

        return cast(Any, jpype.JArray(jpype.JDouble))(y_hat)

    @jpype.JOverride
    def getPredictionForInstance(
        self, instance_or_example: SamoaInstance | Example[SamoaInstance]
    ) -> Prediction:
        raise ProxyNotImplementedError(self, "getPredictionForInstance")

    @jpype.JOverride
    def getConfidenceForPrediction(
        self,
        instance_or_example: SamoaInstance | Example[SamoaInstance],
        prediction: float,
    ) -> float:
        raise ProxyNotImplementedError(self, "getConfidenceForPrediction")

    @jpype.JOverride
    def getSublearners(self) -> MutableSequence[Learner[Example[SamoaInstance]]]:
        raise ProxyNotImplementedError(self, "getSublearners")

    @jpype.JOverride
    def getModel(self) -> MOAObject:
        raise ProxyNotImplementedError(self, "getModel")


@jpype.JImplements(JavaFeatureImportanceClassifier)
@_inherit_overrides
class ProxyJavaFeatureImportanceClassifier(ProxyJavaClassifier):
    """An adapter implementing moa's
    ``moa.learners.featureanalysis.JavaFeatureImportanceClassifier`` interface in
    python.

    https://github.com/Waikato/moa/blob/master/moa/src/main/java/moa/learners/featureanalysis/FeatureImportanceClassifier.java
    """

    @property
    def _py_fi_classifier(self) -> FeatureImportanceClassifier:
        assert self._py_learner is not None, "Python learner is not initialized"
        return self._py_learner

    @jpype.JOverride
    def getFeatureImportances(self, normalize: bool) -> MutableSequence[float]:
        importances = self._py_fi_classifier.get_feature_importances(bool(normalize))
        return cast(Any, jpype.JArray(jpype.JDouble))(importances)

    @jpype.JOverride
    def getTopKFeatures(self, k: int, normalize: bool) -> MutableSequence[int]:
        top_k = self._py_fi_classifier.get_top_k_features(int(k), bool(normalize))
        return cast(Any, jpype.JArray(jpype.JInt))(top_k)


@jpype.JImplements(JavaClassifier, JavaRegressor)
@_inherit_overrides
class ProxyJavaRegressor(ProxyJavaClassifier):
    """An adapter implementing moa's ``moa.learners.classifiers.Regressor`` interface
    in python.

    https://github.com/Waikato/moa/blob/master/moa/src/main/java/moa/classifiers/Regressor.java
    """

    @property
    def _py_regressor(self) -> Regressor:
        assert self._py_learner is not None, "Python learner is not initialized"
        return self._py_learner

    @jpype.JOverride
    def trainOnInstance(
        self, instance_or_example: SamoaInstance | Example[SamoaInstance]
    ) -> None:
        self._py_regressor.train(self._to_py(instance_or_example))

    @jpype.JOverride
    def getVotesForInstance(
        self, instance_or_example: SamoaInstance | Example[SamoaInstance]
    ) -> MutableSequence[float]:
        py_instance = self._to_py(instance_or_example)
        y_hat = self._py_regressor.predict(py_instance)
        votes = [] if y_hat is None else [float(y_hat)]
        return cast(Any, jpype.JArray(jpype.JDouble))(votes)


@jpype.JImplements(JavaClassifier, JavaRegressor, JavaPredictionIntervalLearner)
@_inherit_overrides
class ProxyJavaPredictionIntervalLearner(ProxyJavaRegressor):
    """An adapter implementing moa's
    ``moa.classifiers.predictioninterval.PredictionIntervalLearner`` interface in
    python.

    https://github.com/Waikato/moa/blob/master/moa/src/main/java/moa/classifiers/predictioninterval/PredictionIntervalLearner.java
    """

    @property
    def _py_pi_learner(self) -> PredictionIntervalLearner:
        assert self._py_learner is not None, "Python learner is not initialized"
        return self._py_learner

    @jpype.JOverride
    def getVotesForInstance(
        self, instance_or_example: SamoaInstance | Example[SamoaInstance]
    ) -> MutableSequence[float]:
        py_instance = self._to_py(instance_or_example)
        y_hat = self._py_pi_learner.predict(py_instance)
        votes = [] if y_hat is None else [float(v) for v in y_hat]
        return cast(Any, jpype.JArray(jpype.JDouble))(votes)


@jpype.JImplements(JavaClassifier, JavaSemiSupervisedLearner)
@_inherit_overrides
class ProxyJavaClassifierSSL(ProxyJavaClassifier):
    """An adapter implementing moa's
    ``moa.classifiers.SemiSupervisedLearner`` interface in python.

    https://github.com/Waikato/moa/blob/master/moa/src/main/java/moa/classifiers/SemiSupervisedLearner.java
    """

    @property
    def _py_classifier_ssl(self) -> ClassifierSSL:
        assert self._py_learner is not None, "Python learner is not initialized"
        return self._py_learner

    @jpype.JOverride
    def trainOnUnlabeledInstance(self, instance: SamoaInstance) -> int:
        self._py_classifier_ssl.train_on_unlabeled(self._to_py(instance))
        # MOA learners return -1 here.
        return -1

    @jpype.JOverride
    def addInitialWarmupTrainingInstances(self) -> None:
        raise ProxyNotImplementedError(self, "addInitialWarmupTrainingInstances")


@jpype.JImplements(JavaClassifier, JavaOneClassClassifier)
@_inherit_overrides
class ProxyJavaAnomalyDetector(ProxyJavaClassifier):
    """An adapter implementing moa's
    ``moa.classifiers.OneClassClassifier`` interface in python.

    https://github.com/Waikato/moa/blob/master/moa/src/main/java/moa/classifiers/OneClassClassifier.java
    """

    @property
    def _py_detector(self) -> AnomalyDetector:
        assert self._py_learner is not None, "Python learner is not initialized"
        return self._py_learner

    @jpype.JOverride
    def trainOnInstance(
        self, instance_or_example: SamoaInstance | Example[SamoaInstance]
    ) -> None:
        self._py_detector.train(self._to_py(instance_or_example))

    @jpype.JOverride
    def getVotesForInstance(
        self, instance_or_example: SamoaInstance | Example[SamoaInstance]
    ) -> MutableSequence[float]:
        assert self._schema is not None, "Schema is not set"
        y_hat = self._py_detector.predict(self._to_py(instance_or_example))
        n_classes = self._schema.get_num_classes()
        if y_hat is None:
            votes = np.full(n_classes, 1.0 / n_classes)
        else:
            votes = np.zeros(n_classes)
            votes[y_hat] = 1.0
        return cast(Any, jpype.JArray(jpype.JDouble))(votes)

    @jpype.JOverride
    def getAnomalyScore(self, instance: SamoaInstance) -> float:
        return float(self._py_detector.score_instance(self._to_py(instance)))

    @jpype.JOverride
    def initialize(self, instances: Collection[SamoaInstance]) -> None:
        raise ProxyNotImplementedError(self, "initialize")


@jpype.JImplements(JavaChangeDetector)
@_inherit_overrides
class ProxyJavaDriftDetector(_ProxyOptionHandler):
    """An adapter implementing moa's
    ``moa.classifiers.core.driftdetection.ChangeDetector`` interface in python.

    https://github.com/Waikato/moa/blob/master/moa/src/main/java/moa/classifiers/core/driftdetection/ChangeDetector.java
    """

    def __init__(self, learner_spec: LearnerSpec):
        super().__init__(learner_spec)
        self._py_detector: BaseDriftDetector = self._build()

    def _build(self) -> BaseDriftDetector:
        return learner_from_params(self._learner_spec, None)

    @jpype.JOverride
    def resetLearning(self) -> None:
        self._py_detector = self._build()

    @jpype.JOverride
    def input(self, value: float) -> None:
        self._py_detector.add_element(float(value))

    @jpype.JOverride
    def getChange(self) -> bool:
        return bool(self._py_detector.detected_change())

    @jpype.JOverride
    def getWarningZone(self) -> bool:
        return bool(self._py_detector.detected_warning())

    @jpype.JOverride
    def getEstimation(self) -> float:
        raise ProxyNotImplementedError(self, "getEstimation")

    @jpype.JOverride
    def getDelay(self) -> float:
        raise ProxyNotImplementedError(self, "getDelay")

    @jpype.JOverride
    def getOutput(self) -> MutableSequence[float]:
        raise ProxyNotImplementedError(self, "getOutput")

    @jpype.JOverride
    def copy(self) -> JavaChangeDetector:
        return ProxyJavaDriftDetector(self._learner_spec)

    @jpype.JOverride
    def prepareForUse(
        self,
        monitor: TaskMonitor | None = None,
        repository: ObjectRepository | None = None,
    ) -> None:
        self._py_detector = self._build()


@jpype.JImplements(JavaClusterer)
@_inherit_overrides
class ProxyJavaClusterer(_ProxyLearner):
    """An adapter implementing moa's ``moa.clusterers.Clusterer`` interface in python.

    https://github.com/Waikato/moa/blob/master/moa/src/main/java/moa/clusterers/Clusterer.java
    """

    @property
    def _py_clusterer(self) -> Clusterer:
        assert self._py_learner is not None, "Python learner is not initialized"
        return self._py_learner

    @jpype.JOverride
    def trainOnInstance(self, instance: SamoaInstance) -> None:
        self._py_clusterer.train(self._to_py(instance))

    @staticmethod
    def _to_java_clustering(result: ClusteringResult) -> Clustering:
        """Convert a python clustering result to a MOA clustering.

        MOA cluster centers have one more value than the features, for the class.
        :class:`~capymoa.base.MOAClusterer` drops it, so we add a ``0.0`` here.
        """
        clustering = Clustering()
        for center, weight, radius in zip(
            result.get_centers(), result.get_weights(), result.get_radii()
        ):
            java_center = jpype.JArray(jpype.JDouble)([*center, 0.0])
            clustering.add(SphereCluster(java_center, float(radius), float(weight)))
        return clustering

    @jpype.JOverride
    def getClusteringResult(self) -> Clustering:
        return self._to_java_clustering(self._py_clusterer.get_clustering_result())

    @jpype.JOverride
    def getMicroClusteringResult(self) -> Clustering:
        return self._to_java_clustering(
            self._py_clusterer.get_micro_clustering_result()
        )

    @jpype.JOverride
    def implementsMicroClusterer(self) -> bool:
        return bool(self._py_clusterer.implements_micro_clusters())

    @jpype.JOverride
    def keepClassLabel(self) -> bool:
        return False

    @jpype.JOverride
    def getSubClusterers(self) -> MutableSequence[JavaClusterer]:
        raise ProxyNotImplementedError(self, "getSubClusterers")

    @jpype.JOverride
    def getVotesForInstance(self, instance: SamoaInstance) -> MutableSequence[float]:
        raise ProxyNotImplementedError(self, "getVotesForInstance")
