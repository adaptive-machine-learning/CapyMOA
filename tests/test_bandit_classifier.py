import json
import tempfile

import pytest

from capymoa.automl import BanditClassifier, EpsilonGreedy
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
    from capymoa.classifier import HoeffdingTree, NaiveBayes
    from capymoa.core.io import load_model, save_model

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


def test_classification_evaluator_save_load_roundtrip():
    """A bare ClassificationEvaluator must round-trip through save_model/load_model.

    This is the smallest object that reproduces the recursion, so it pins the
    root cause rather than the BanditClassifier that happens to hold one.
    """
    from capymoa.core.io import load_model, save_model
    from capymoa.evaluation.evaluation import ClassificationEvaluator

    stream = SEA()
    evaluator = ClassificationEvaluator(schema=stream.get_schema())
    for _ in range(100):
        evaluator.update(stream.next_instance().y_index, 0)

    with tempfile.TemporaryFile() as fd:
        save_model(evaluator, fd)
        fd.seek(0)
        restored = load_model(fd)

    assert isinstance(restored, ClassificationEvaluator)
    assert restored.instances_seen == evaluator.instances_seen
    assert restored.accuracy() == pytest.approx(evaluator.accuracy())


def test_evaluator_getattr_does_not_recurse_before_init():
    """__getattr__ must raise AttributeError, not call itself, on a bare object.

    Python asks for __setstate__ on an object whose __init__ has not run. If
    __getattr__ reads an attribute that is not set yet, it lands back in
    __getattr__ and the stack runs out. Reproduced here without any unpickling,
    which keeps the test from needing to observe a RecursionError to fail.
    """
    from capymoa.evaluation.evaluation import (
        ClassificationEvaluator,
        ClassificationWindowedEvaluator,
    )
    from capymoa.evaluation.results import PrequentialResults

    for cls in (
        ClassificationEvaluator,
        ClassificationWindowedEvaluator,
        PrequentialResults,
    ):
        bare = cls.__new__(cls)
        # hasattr is the probe pickle makes. It swallows AttributeError, so it
        # only returns if __getattr__ declined the name; if __getattr__ recursed
        # instead, the RecursionError escapes and the test fails.
        assert not hasattr(bare, "__setstate__")
