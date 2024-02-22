from capymoa.datasets.datasets import ElectricityTiny, CovtypeTiny
from capymoa.learner.classifier.CPSSDS import CPSSDS
from test_utility.ssl_helpers import assert_ssl_evaluation
import pytest


@pytest.mark.parametrize(
    "learner, stream, expectation",
    [
        ("NaiveBayes", ElectricityTiny(), 76.6),
        ("HoeffdingTree", ElectricityTiny(), 66.2),
        ("NaiveBayes", CovtypeTiny(), 55.7),
        ("HoeffdingTree", CovtypeTiny(), 53.3),
    ],
    ids=[
        "ElectricityTiny-NaiveBayes",
        "ElectricityTiny-HoeffdingTree",
        "CovtypeTiny-NaiveBayes",
        "CovtypeTiny-HoeffdingTree",
    ],
)
def test_CPSSDS(learner, stream, expectation):
    assert_ssl_evaluation(
        CPSSDS(learner, 100, schema=stream.schema),
        stream,
        expectation,
        label_probability=0.5,
    )
