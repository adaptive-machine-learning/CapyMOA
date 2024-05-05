from capymoa.evaluation import AUCEvaluator
from capymoa.anomaly import (
    HalfSpaceTrees,
)
from capymoa.base import Classifier
from capymoa.base import MOAClassifier
from capymoa.datasets import ElectricityTiny
import pytest
from functools import partial
from typing import Callable, Optional
from capymoa.base import _extract_moa_learner_CLI

from capymoa.stream._stream import Schema


@pytest.mark.parametrize(
    "learner_constructor,auc,cli_string",
    [
        (partial(HalfSpaceTrees, window_size=100, number_of_trees=25, max_depth=15), 0.62, None),
    ],
    ids=[
        "HalfSpaceTrees",
    ],
)
def test_anomaly_detectors(
    learner_constructor: Callable[[Schema], Classifier],
    auc: float,
    cli_string: Optional[str],
):
    """Test on tiny is a fast running simple test to check if a learner's
    performance has changed.

    Notice how we use the `partial` function to creates a new function with
    hyperparameters already set. This allows us to use the same test function
    for different learners with different hyperparameters.

    :param learner_constructor: A partially applied constructor for the learner
    :param auc: Expected AUC score
    :param cli_string: Expected CLI string for the learner or None
    """
    stream = ElectricityTiny()
    evaluator = AUCEvaluator(schema=stream.get_schema())

    learner: Classifier = learner_constructor(schema=stream.get_schema())

    while stream.has_more_instances():
        instance = stream.next_instance()
        proba = learner.predict_proba(instance)
        evaluator.update(instance.y_index, proba)
        learner.train(instance)

    # Check if the AUC score matches the expected value for both evaluator types
    actual_auc = evaluator.auc()
    assert actual_auc == pytest.approx(
        auc, abs=0.1
    ), f"Basic Eval: Expected accuracy of {auc:0.1f} got {actual_auc: 0.1f}"

    # Optionally check the CLI string if it was provided
    if isinstance(learner, MOAClassifier) and cli_string is not None:
        cli_str = _extract_moa_learner_CLI(learner).strip("()")
        assert cli_str == cli_string, "CLI does not match expected value"
