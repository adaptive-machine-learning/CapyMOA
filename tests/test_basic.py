from capymoa.evaluation import ClassificationEvaluator
from capymoa.learner.classifier.classifiers import OnlineBagging
from capymoa.datasets import ElectricityTiny
import pytest


def test_basic_classification():
    """Test the basic classification functionality."""
    stream = ElectricityTiny()
    learner = OnlineBagging(schema=stream.get_schema(), ensemble_size=5)
    evaluator = ClassificationEvaluator(schema=stream.get_schema())

    while stream.has_more_instances():
        instance = stream.next_instance()
        prediction = learner.predict(instance)
        evaluator.update(instance.y_label, prediction)
        learner.train(instance)

    assert evaluator.accuracy() == pytest.approx(84.6)
