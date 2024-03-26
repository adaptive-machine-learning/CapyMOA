from capymoa.datasets.datasets import ElectricityTiny, CovtypeTiny
from capymoa.learner.classifier import EFDT
from test_utility.ssl_helpers import assert_ssl_evaluation
import pytest


@pytest.mark.parametrize(
    "stream, expectation",
    [
        (ElectricityTiny(), 47.5),
        (CovtypeTiny(), 53.2),
    ],
    ids=["ElectricityTiny", "CovtypeTiny"]
)
def test_EFDT(stream, expectation):
    # The optimizer steps are set to 10 to speed up the test
    learner = EFDT(schema=stream.schema)
    assert_ssl_evaluation(
        learner,
        stream,
        expectation,
    )
