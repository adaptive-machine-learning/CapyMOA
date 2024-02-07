import pytest
from capymoa.evaluation.evaluation import prequential_SSL_evaluation
from capymoa.learner import ClassifierSSL
from capymoa.stream import Stream

def assert_ssl_evaluation(
    learner: ClassifierSSL,
    stream: Stream,
    expectation: float,
    label_probability: float = 0.01,
    max_instances: int = 1000,
):
    results = prequential_SSL_evaluation(
        stream=stream,
        learner=learner,
        label_probability=label_probability,
        window_size=10,
        optimise=False,
        max_instances=max_instances,
    )

    assert results["cumulative"].accuracy() == pytest.approx(expectation), \
        f"Expected accuracy of {expectation} but got {results['cumulative'].accuracy()}" + \
        f" for learner {learner} on stream {stream}"
