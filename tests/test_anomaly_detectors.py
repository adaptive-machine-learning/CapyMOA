from collections.abc import Callable
from functools import partial

import pytest

from capymoa._cli import cli_str_classifier
from capymoa.anomaly import (
    AdaptiveIsolationForest,
    Autoencoder,
    HalfSpaceTrees,
    IForestASD,
    Loda,
    OnlineIsolationForest,
    RobustRandomCutForest,
    RSHash,
    StreamingIsolationForest,
    StreamRHF,
)
from capymoa.base import AnomalyDetector, MOAClassifier
from capymoa.datasets import ElectricityTiny
from capymoa.evaluation import AnomalyDetectionEvaluator
from capymoa.instance import Optional
from capymoa.stream._stream import Schema


@pytest.mark.parametrize(
    "learner_constructor,auc,cli_string",
    [
        (
            partial(HalfSpaceTrees, window_size=100, number_of_trees=25, max_depth=15),
            0.54,
            None,
        ),
        (
            partial(
                OnlineIsolationForest,
                window_size=100,
                num_trees=32,
                max_leaf_samples=32,
            ),
            0.42,
            None,
        ),
        (
            partial(Autoencoder, hidden_layer=2, learning_rate=0.5, threshold=0.6),
            0.57,
            None,
        ),
        (partial(StreamRHF, num_trees=5, max_height=3), 0.72, None),
        (
            partial(
                StreamingIsolationForest,
                window_size=256,
                n_trees=100,
                height=None,
                seed=42,
            ),
            0.60,
            None,
        ),
        (
            partial(
                RobustRandomCutForest,
                tree_size=50,
                n_trees=10,
                random_state=42,
            ),
            0.54,
            None,
        ),
        (
            partial(
                AdaptiveIsolationForest,
                window_size=256,
                n_trees=100,
                height=None,
                seed=42,
                m_trees=1,
                weights=0.5,
            ),
            0.83,
            None,
        ),
        (
            partial(
                IForestASD,
                window_size=256,
                sample_size=64,
                n_trees=100,
                height_limit=None,
                random_state=42,
            ),
            0.61,
            None,
        ),
        (
            partial(
                Loda,
                n_projections=10,
                window_size=100,
                random_state=42,
            ),
            0.65,
            None,
        ),
        (
            partial(RSHash, m=300, s=256, w=4, p=10000, seed=42),
            # Measured at 0.61999 in isolation, drifting to ~0.62006 depending
            # on what ran earlier in the session. 0.61 put the top of the
            # tolerance at exactly 0.62, so the test failed intermittently.
            0.62,
            None,
        ),
    ],
    ids=[
        "HalfSpaceTrees",
        "OnlineIsolationForest",
        "Autoencoder",
        "StreamRHF",
        "StreamingIsolationForest",
        "RobustRandomCutForest",
        "AdaptiveIsolationForest",
        "IForestASD",
        "Loda",
        "RSHash",
    ],
)
def test_anomaly_detectors(
    learner_constructor: Callable[[Schema], AnomalyDetector],
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
    evaluator = AnomalyDetectionEvaluator(schema=stream.get_schema())

    learner: AnomalyDetector = learner_constructor(schema=stream.get_schema())

    for instance in stream:
        score = learner.score_instance(instance)
        evaluator.update(instance.y_index, score)
        learner.train(instance)

    # Check if the AUC score matches the expected value for both evaluator types
    actual_auc = evaluator.auc()
    assert actual_auc == pytest.approx(auc, abs=0.01), (
        f"Basic Eval: Expected accuracy of {auc:0.1f} got {actual_auc: 0.01f}"
    )

    # Optionally check the CLI string if it was provided
    if isinstance(learner, MOAClassifier) and cli_string is not None:
        cli_str = cli_str_classifier(learner).strip("()")
        assert cli_str == cli_string, "CLI does not match expected value"
