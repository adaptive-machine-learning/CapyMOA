from capymoa.datasets.datasets import ElectricityTiny, CovtypeTiny
from capymoa.learner.classifier import HoeffdingTree
from test_utility.ssl_helpers import assert_ssl_evaluation
import pytest


@pytest.mark.parametrize(
    "stream, expectation",
    [
        (ElectricityTiny(), 46.0),
        (CovtypeTiny(), 46.0),
    ],
    ids=["ElectricityTiny", "CovtypeTiny"]
)
def test_HT(stream, expectation):
    # The optimizer steps are set to 10 to speed up the test
    learner = HoeffdingTree(
        schema=stream.schema,
        grace_period=201,
        # split_criterion="gini",
        confidence=1e-3,
        tie_threshold=0.055,
        # leaf_prediction="mc",
        nb_threshold=1,
        # numeric_attribute_observer="FIMTDDNumericAttributeClassObserver",
        binary_split=True,
        min_branch_fraction=0.02,
        max_share_to_split=0.98,
        max_byte_size=33554434,
        memory_estimate_period=1000001,
        stop_mem_management=True,
        remove_poor_attrs=True,
        disable_prepruning=False,
    )
    assert_ssl_evaluation(
        learner,
        stream,
        expectation,
    )
