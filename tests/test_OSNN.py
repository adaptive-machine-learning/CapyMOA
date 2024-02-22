from capymoa.datasets.datasets import ElectricityTiny, CovtypeTiny
from test_utility.ssl_helpers import assert_ssl_evaluation
import pytest
import importlib

@pytest.mark.parametrize(
    "stream, expectation",
    [
        (ElectricityTiny(), 46.1),
        (CovtypeTiny(), 26.3),
    ],
    ids=["ElectricityTiny", "CovtypeTiny"],
)
def test_OSNN(stream, expectation):
    pytest.importorskip("torch.nn", reason="PyTorch not installed. Skipping test.")
    OSNN = importlib.import_module("capymoa.learner.classifier.OSNN").OSNN
    # The optimizer steps are set to 10 to speed up the test
    learner = OSNN(optim_steps=10)
    assert_ssl_evaluation(
        learner,
        stream,
        expectation,
    )
