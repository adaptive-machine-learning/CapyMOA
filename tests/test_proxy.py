import jpype
from moa.classifiers.core.driftdetection import ChangeDetector as JavaChangeDetector
from moa.learners.featureanalysis import (
    FeatureImportanceClassifier as JavaFeatureImportanceClassifier,
)

from capymoa.anomaly import evaluate_anomaly
from capymoa.base import (
    MOAAnomalyDetector,
    MOAClassifier,
    MOAClassifierSSL,
    MOAClusterer,
    MOAPredictionIntervalLearner,
    MOARegressor,
)
from capymoa.classifier import evaluate_classifier
from capymoa.cluster import ClusTree
from capymoa.core.moa.proxy import (
    ProxyJavaAnomalyDetector,
    ProxyJavaClassifier,
    ProxyJavaClassifierSSL,
    ProxyJavaClusterer,
    ProxyJavaDriftDetector,
    ProxyJavaFeatureImportanceClassifier,
    ProxyJavaPredictionIntervalLearner,
    ProxyJavaRegressor,
)
from capymoa.datasets import ElectricityTiny, FriedTiny
from capymoa.drift.detectors import ADWIN
from capymoa.feature import MOAFeatureImportanceClassifier
from capymoa.regressor import evaluate_regressor
from capymoa.ssl import evaluate_ssl
from capymoa.uncertainty import evaluate_prediction_interval


def test_proxy_java_classifier():
    stream = ElectricityTiny()
    proxy = MOAClassifier(
        ProxyJavaClassifier({"learner": "capymoa.classifier.SGDClassifier"}),
        stream.schema,
    )
    evaluate_classifier(stream, proxy, optimise=False)


class _ProxyClusterer(MOAClusterer):
    def implements_micro_clusters(self) -> bool:
        return bool(self.moa_learner.implementsMicroClusterer())

    def implements_macro_clusters(self) -> bool:
        return False


def test_proxy_java_clusterer():
    """The proxy gives the same micro-clusters as the python clusterer."""
    stream = ElectricityTiny()
    proxy = _ProxyClusterer(
        ProxyJavaClusterer({"learner": "capymoa.cluster.ClusTree"}), stream.schema
    )
    clusterer = ClusTree(stream.schema)
    for _ in range(500):
        instance = stream.next_instance()
        proxy.train(instance)
        clusterer.train(instance)

    expected = clusterer.get_micro_clustering_result()
    actual = proxy.get_micro_clustering_result()
    assert len(expected.get_centers()) > 0
    assert actual.get_centers() == expected.get_centers()
    assert actual.get_weights() == expected.get_weights()
    assert actual.get_radii() == expected.get_radii()
    assert str(proxy) == "ClusTree"


def test_proxy_java_regressor():
    stream = FriedTiny()
    proxy = MOARegressor(
        stream.schema,
        moa_learner=ProxyJavaRegressor({"learner": "capymoa.regressor.SGDRegressor"}),
    )
    evaluate_regressor(stream, proxy, optimise=False)


def test_proxy_java_drift_detector():
    """The proxy detects the same drifts as the python detector."""
    spec = {"learner": "capymoa.drift.detectors.adwin.ADWIN"}
    proxy = jpype.JObject(ProxyJavaDriftDetector(spec), JavaChangeDetector)
    proxy.prepareForUse()
    detector = ADWIN()

    data = [0.0] * 1000 + [1.0] * 1000
    proxy_changes, py_changes = [], []
    for i, x in enumerate(data):
        proxy.input(x)
        detector.add_element(x)
        if proxy.getChange():
            proxy_changes.append(i)
        if detector.detected_change():
            py_changes.append(i)

    assert len(py_changes) > 0
    assert proxy_changes == py_changes
    assert proxy.copy() is not None


def test_proxy_java_anomaly_detector():
    stream = ElectricityTiny()
    proxy = MOAAnomalyDetector(
        stream.schema,
        moa_learner=ProxyJavaAnomalyDetector({"learner": "capymoa.anomaly.Loda"}),
        random_seed=1,
    )
    evaluate_anomaly(stream, proxy, optimise=False)


def test_proxy_java_classifier_ssl():
    stream = ElectricityTiny()
    proxy = MOAClassifierSSL(
        ProxyJavaClassifierSSL({"learner": "capymoa.ssl.SLEADE"}),
        stream.schema,
    )
    evaluate_ssl(stream, proxy, label_probability=0.5, optimise=False)


def test_proxy_java_prediction_interval_learner():
    stream = FriedTiny()
    proxy = MOAPredictionIntervalLearner(
        stream.schema,
        moa_learner=ProxyJavaPredictionIntervalLearner(
            {"learner": "capymoa.uncertainty.MVE"}
        ),
    )
    evaluate_prediction_interval(stream, proxy, optimise=False)
    stream.restart()
    lower, mean, upper = proxy.predict(stream.next_instance())
    assert lower <= mean <= upper


def test_proxy_java_feature_importance_classifier():
    """The proxy gives the same importances as the python classifier."""
    stream = ElectricityTiny()
    spec = {"learner": "capymoa.feature.MOAFeatureImportanceClassifier"}
    proxy = jpype.JObject(
        ProxyJavaFeatureImportanceClassifier(spec), JavaFeatureImportanceClassifier
    )
    proxy.setModelContext(stream.schema.get_moa_header())
    proxy.prepareForUse()
    classifier = MOAFeatureImportanceClassifier(stream.schema)
    for _ in range(500):
        instance = stream.next_instance()
        proxy.trainOnInstance(instance.java_instance)
        classifier.train(instance)

    expected = classifier.get_feature_importances()
    assert sum(expected) > 0
    assert list(proxy.getFeatureImportances(True)) == expected
    assert list(proxy.getTopKFeatures(2, True)) == classifier.get_top_k_features(2)
