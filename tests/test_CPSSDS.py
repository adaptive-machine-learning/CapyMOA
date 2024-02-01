from capymoa.datasets.datasets import ElectricityTiny, CovtypeTiny
from capymoa.learner.classifier.CPSSDS import CPSSDS
from test_utility.ssl_helpers import assert_ssl_evaluation
import pytest

@pytest.mark.parametrize(
    "learner, stream, expectation", 
    [
        ("NaiveBayes", ElectricityTiny(), 70.0),
        ("HoeffdingTree", ElectricityTiny(), 59.60),
        ("NaiveBayes", CovtypeTiny(), 54.6),
        ("HoeffdingTree", CovtypeTiny(), 52.2),
    ],
    ids=[
        "ElectricityTiny-NaiveBayes", 
        "ElectricityTiny-HoeffdingTree", 
        "CovtypeTiny-NaiveBayes", 
        "CovtypeTiny-HoeffdingTree"
    ]
)
def test_CPSSDS(learner, stream, expectation):
    assert_ssl_evaluation(
        CPSSDS(learner, 100, schema=stream.schema),
        stream,
        expectation,
        label_probability=0.5
    )
