import pytest

from capymoa.base import Clusterer
from capymoa.cluster import Clustream, Clustream_with_kmeans, evaluate_cluster
from capymoa.cluster.evaluate import ClusteringEvaluator
from capymoa.evaluation import prequential_evaluation
from capymoa.stream.generator import RandomRBFGeneratorDrift


def _stream():
    return RandomRBFGeneratorDrift(number_of_attributes=2)


def test_evaluate_cluster_keys_and_counts():
    stream = _stream()
    clusterer = Clustream(schema=stream.get_schema())
    results = evaluate_cluster(
        stream, clusterer, max_instances=250, update_interval=100
    )
    assert results["instances"] == 250
    assert {"learner", "stream", "instances", "macro", "micro"} <= set(results)
    assert 0 < len(results["micro"]) <= 3
    # Clustream has no macro clusters, so the list is empty
    assert results["macro"] == []


def test_evaluate_cluster_rejects_bad_interval():
    stream = _stream()
    clusterer = Clustream(schema=stream.get_schema())
    with pytest.raises(ValueError):
        evaluate_cluster(stream, clusterer, update_interval=0)


def test_evaluate_cluster_macro_snapshots():
    stream = _stream()
    clusterer = Clustream_with_kmeans(schema=stream.get_schema(), k_option=3)
    assert clusterer.implements_macro_clusters()
    results = evaluate_cluster(
        stream, clusterer, max_instances=300, update_interval=100
    )
    assert results["instances"] == 300
    assert len(results["macro"]) > 0


def test_clusterer_evaluator_records_snapshots():
    stream = _stream()
    clusterer = Clustream(schema=stream.get_schema())
    evaluator = ClusteringEvaluator(update_interval=10)
    for _ in range(20):
        clusterer.train(stream.next_instance())
        evaluator.update(clusterer)
    assert evaluator.get_instances_seen() == 20


def test_prequential_dispatches_clusterer():
    stream = _stream()
    clusterer = Clustream(schema=stream.get_schema())
    assert isinstance(clusterer, Clusterer)
    results = prequential_evaluation(
        stream, clusterer, max_instances=100, update_interval=50
    )
    assert results["instances"] == 100
    assert "micro" in results
