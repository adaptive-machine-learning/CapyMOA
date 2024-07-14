from capymoa.datasets._datasets import ElectricityTiny, CovtypeTiny
from capymoa.ssl.classifier import OSNN

import pytest
from capymoa.evaluation.evaluation import prequential_ssl_evaluation
from capymoa.base import ClassifierSSL
from capymoa.stream import Stream
from functools import partial


def assert_ssl_evaluation(
    learner: ClassifierSSL,
    stream: Stream,
    expectation: float,
    label_probability: float = 0.01,
    max_instances: int = 1000,
):
    results = prequential_ssl_evaluation(
        stream=stream,
        learner=learner,
        label_probability=label_probability,
        window_size=10,
        max_instances=max_instances,
    )

    assert results['cumulative'].accuracy() == pytest.approx(expectation), (
        f"Expected accuracy of {expectation} but got {results['cumulative'].accuracy()}"
        + f" for learner {learner} on stream {stream}"
    )

@pytest.mark.parametrize(
    "learner_constructor, stream_constructor, expectation, label_probability",
    [
        (partial(OSNN, optim_steps=10), ElectricityTiny, 46.1, None),
        (partial(OSNN, optim_steps=10), CovtypeTiny, 26.3, None),
    ],
    ids=[
        "OSNN_ElectricityTiny",
        "OSNN_CovtypeTiny",
    ],
)
def test_ssl_classifiers(learner_constructor, stream_constructor, expectation, label_probability):
    # The optimizer steps are set to 10 to speed up the test
    stream = stream_constructor()
    learner = learner_constructor(schema=stream.get_schema())

    if label_probability is None:
        label_probability = 0.01

    assert_ssl_evaluation(
        learner,
        stream,
        expectation,
        label_probability=label_probability,
    )
