from capymoa.datasets.datasets import ElectricityTiny, CovtypeTiny
from capymoa.learner.classifier.OSNN import OSNN
from test_utility.ssl_helpers import assert_ssl_evaluation
import pytest

@pytest.mark.parametrize(
    "stream, expectation", 
    [
        (ElectricityTiny(), 35.5),
        (CovtypeTiny(), 22.0),
    ],
    ids=["ElectricityTiny", "CovtypeTiny"]
)
def test_OSNN(stream, expectation):
    # The optimizer steps are set to 10 to speed up the test
    learner = OSNN(optim_steps=10)
    assert_ssl_evaluation(
        learner,
        stream,
        expectation,
    )
