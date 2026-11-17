import json
import tempfile

import pytest

from capymoa.automl import BanditClassifier, EpsilonGreedy
from capymoa.classifier import HoeffdingTree, NaiveBayes
from capymoa.core.io import load_model, save_model
from capymoa.core.moa._cli import cli_str_classifier
from capymoa.stream.generator import SEA


def test_bandit_classifier_parameter_initialization():
    """Test that BanditClassifier passes parameters through the constructor."""

    # Create a config file with a HoeffdingTree(grace_period=50)
    config = {
        "algorithms": [
            {
                "algorithm": "HoeffdingTree",
                "parameters": [
                    {"parameter": "grace_period", "value": 50},
                ],
            }
        ]
    }

    with tempfile.NamedTemporaryFile("w", delete=False) as f:
        json.dump(config, f)
        config_path = f.name

    # Build a small stream and schema
    stream = SEA()
    schema = stream.get_schema()

    # Instantiate BanditClassifier from config
    bc = BanditClassifier(
        schema=schema,
        config_file=config_path,
        policy=EpsilonGreedy(epsilon=0.1, burn_in=1),
    )

    # Retrieve the created model
    model = bc.active_models[0]

    # Check that the constructor parameter was applied (CLI should include "-g 50")
    cli_str = cli_str_classifier(model)
    assert "-g 50" in cli_str, f"Expected grace_period=50, got CLI: {cli_str}"

    # Basic functional check
    instance = next(iter(stream))
    model.predict(instance)
    model.train(instance)


def test_bandit_classifier_save_load_roundtrip():
    """A saved BanditClassifier must load, and load with its state intact.

    The classifier owns a list of ClassificationEvaluator objects, and their
    __getattr__ used to recurse into itself while unpickling, so save_model
    succeeded but load_model died with RecursionError.
    """
    stream = SEA()
    schema = stream.get_schema()
    learner = BanditClassifier(
        schema=schema,
        random_seed=42,
        base_classifiers=[HoeffdingTree, NaiveBayes],
        policy=EpsilonGreedy(epsilon=0.1, burn_in=100),
    )
    for _ in range(200):
        learner.train(stream.next_instance())

    with tempfile.TemporaryFile() as fd:
        save_model(learner, fd)
        fd.seek(0)
        restored: BanditClassifier = load_model(fd)

    assert isinstance(restored, BanditClassifier)
    assert [type(m).__name__ for m in restored.active_models] == [
        type(m).__name__ for m in learner.active_models
    ]
    assert restored.log_cnt == learner.log_cnt

    # The restored model must keep predicting what the original predicts, which
    # only holds if the evaluators and the models behind them both came back.
    original_stream, restored_stream = SEA(), SEA()
    for _ in range(200):
        original_stream.next_instance()
        restored_stream.next_instance()

    for _ in range(20):
        assert learner.predict(original_stream.next_instance()) == restored.predict(
            restored_stream.next_instance()
        )

    # Evaluator state is part of the graph too, not just the models.
    assert restored.metrics[0].accuracy() == pytest.approx(
        learner.metrics[0].accuracy()
    )
