"""Test harness for classifiers in the CapyMOA framework.

Add new methods to `tests/resources/classifier.yml`. Review `Case` for the required
structure.
"""

from dataclasses import dataclass
from itertools import islice
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import yaml

from capymoa.base import Classifier, LearnerSpec, MOAClassifier, learner_from_params
from capymoa.core.io import load_model, save_model
from capymoa.core.moa._cli import cli_str_classifier
from capymoa.datasets import ElectricityTiny
from capymoa.evaluation import prequential_evaluation
from capymoa.evaluation.evaluation import _is_fast_mode_compilable
from capymoa.stream import Schema
from capymoa.stream.generator import RandomTreeGenerator

WINDOW_SIZE = 100
RANDOM_SEED = 1


@dataclass
class Case:
    id: str
    """The unique identifier of the test case."""

    learner: str
    """The fully qualified class name of the learner."""

    params: dict[str, Any]
    """The learner's parameters as a dictionary."""

    accuracy: float
    """The expected cumulative accuracy of the learner."""

    win_accuracy: float
    """The expected windowed accuracy of the learner."""

    cli_string: str | None = None
    """If set check the cli string matches."""

    always_deterministic: bool = False
    """If set assume the learner is always deterministic in `test_determinism`."""

    skip_test_determinism: str | None = None
    """If set skip the `test_determinism` test with the given reason."""

    skip_test_predict: str | None = None
    """If set skip the `test_predict` test with the given reason."""

    skip_test_save_then_load: str | None = None
    """If set skip the `test_save_then_load` test with the given reason."""

    skip_test_optimise: str | None = None
    """If set skip the `test_optimise` test with the given reason."""

    length_test_determinism: int = 50
    """Number of stream instances to replay in the determinism test."""

    needs_torch: bool = False
    """If set this learner requires PyTorch. Deselected by `-m "not torch"`."""

    predicts_before_training: bool = False
    """If set this learner legitimately returns a real (non-`None`) prediction
    before any `train()` call (e.g. a randomly-initialised neural network),
    instead of abstaining like most learners."""

    batch_size: int = 1
    """Batch size used by `test_accuracy`/`test_optimise`, for learners that
    support mini-batches (e.g. `Finetune`)."""

    def new_learner(self, schema: Schema, random_seed: int = RANDOM_SEED) -> Classifier:
        spec = LearnerSpec(learner=self.learner, params=self.params)
        return learner_from_params(spec, schema, random_seed=random_seed)


def _load_cases() -> list[Case]:
    path = Path(__file__).parent / "resources" / "classifier.yml"
    with open(path, "r") as f:
        return [Case(**c) for c in yaml.safe_load(f)]


CASES = _load_cases()


def _probas_close(a: np.ndarray | None, b: np.ndarray | None) -> bool:
    """Return True if both are None or both are numerically close."""
    if a is None or b is None:
        return a is None and b is None
    return bool(np.allclose(a, b))


@pytest.mark.parametrize(
    "case",
    [pytest.param(c, marks=pytest.mark.torch) if c.needs_torch else c for c in CASES],
    ids=[c.id for c in CASES],
)
class TestClassifier:
    def test_accuracy(self, case: Case):
        """Check ``accuracy`` has not changed from expectation. Not a benchmark."""
        stream = ElectricityTiny()
        learner = case.new_learner(stream.schema)
        results = prequential_evaluation(
            stream, learner, window_size=WINDOW_SIZE, batch_size=case.batch_size
        )
        params = learner.get_params()

        missing = {
            k: (v, params.get(k, "<missing>"))
            for k, v in case.params.items()
            if k not in params or params[k] != v
        }
        assert not missing, (
            "Provided parameters must be a subset of the resolved parameters. "
            f"Mismatches (expected, got): {missing}"
        )

        # Optionally check the CLI string if it was provided
        if case.cli_string is not None:
            assert isinstance(learner, MOAClassifier), (
                "Wrong type, only MOA learners have CLI strings, "
                f"expected `MOAClassifier` but got `{type(learner)}`."
            )
            cli_str = cli_str_classifier(learner)
            assert cli_str == case.cli_string, (
                "CLI does not match expected value, "
                f"expected `{case.cli_string}` but got `{cli_str}`."
            )

        # Check accuracy.
        accuracy = results.cumulative.accuracy()  # type: ignore
        win_accuracy = results.windowed.accuracy()[-1]  # type: ignore
        assert accuracy == pytest.approx(case.accuracy), (
            "Unexpected cumulative accuracy, "
            f"expected {case.accuracy} but got {accuracy}."
        )
        assert win_accuracy == pytest.approx(case.win_accuracy), (
            "Unexpected windowed accuracy, "
            f"expected {case.win_accuracy} but got {win_accuracy}."
        )

    def test_optimise(self, case: Case):
        """Check fast and slow prequential loops return the same results."""
        if case.skip_test_optimise is not None:
            pytest.skip(case.skip_test_optimise)

        stream = ElectricityTiny()
        fast_learner = case.new_learner(stream.schema)
        if not _is_fast_mode_compilable(stream, fast_learner):
            pytest.skip("Learner does not support the fast prequential loop.")

        fast_results = prequential_evaluation(
            stream, fast_learner, window_size=WINDOW_SIZE, batch_size=case.batch_size
        )
        fast_accuracy = fast_results.cumulative.accuracy()  # type: ignore
        fast_win_accuracy = fast_results.windowed.accuracy()[-1]  # type: ignore

        slow_learner = case.new_learner(stream.schema)
        slow_results = prequential_evaluation(
            stream,
            slow_learner,
            window_size=WINDOW_SIZE,
            optimise=False,
            batch_size=case.batch_size,
        )
        slow_accuracy = slow_results.cumulative.accuracy()  # type: ignore
        slow_win_accuracy = slow_results.windowed.accuracy()[-1]  # type: ignore
        assert slow_accuracy == pytest.approx(fast_accuracy), (
            "Slow (optimise=False) loop cumulative accuracy does not match "
            f"the fast loop, expected {fast_accuracy} but got {slow_accuracy}."
        )
        assert slow_win_accuracy == pytest.approx(fast_win_accuracy), (
            "Slow (optimise=False) loop windowed accuracy does not match "
            f"the fast loop, expected {fast_win_accuracy} but got {slow_win_accuracy}."
        )

    def test_predict(self, case: Case):
        """Check ``predict`` and ``predict_proba`` behave as expected."""
        if case.skip_test_predict is not None:
            pytest.skip(case.skip_test_predict)

        n_classes = 10
        length = 50
        did_predict = False

        stream = RandomTreeGenerator(num_classes=n_classes)
        learner = case.new_learner(stream.get_schema())

        # Learner must abstain before it is trained, unless it's declared to
        # legitimately predict beforehand (e.g. a randomly-initialised NN).
        first_instance = next(stream)
        if case.predicts_before_training:
            assert (
                learner.predict(first_instance) is not None
                and learner.predict_proba(first_instance) is not None
            ), "Expected prediction before training."
        else:
            assert (
                learner.predict(first_instance) is None
                and learner.predict_proba(first_instance) is None
            ), "Expected abstention before training."

        # Prequential loop (predict, then train)
        for instance in islice(stream, length):
            # Check probabilities are as expected (on unseen instance)
            y_pred = learner.predict(instance)
            y_proba = learner.predict_proba(instance)
            learner.train(instance)

            if y_proba is not None and y_pred is not None:
                did_predict = True

                assert isinstance(y_proba, np.ndarray), (
                    f"Wrong type, expected `np.ndarray` but got `{type(y_proba)}`"
                )
                assert y_proba.dtype == np.float64, (
                    f"Wrong dtype, expected `np.float64` but got `{y_proba.dtype}`"
                )
                # `predict_proba` must always be shaped by the schema, some methods
                # incorrectly use observed classes.
                # (https://github.com/adaptive-machine-learning/backlog/issues/96).
                assert y_proba.shape == (n_classes,), (
                    f"Wrong shape, expected ({n_classes},) but got `{y_proba.shape}`"
                )
                assert y_proba.sum() == pytest.approx(1.0), (
                    f"Probabilities must sum to 1, expected 1 but got `{y_proba.sum()}`"
                )
                # `predict` must return a built-in `int`, not `numpy.int64`.
                assert isinstance(y_pred, int), (
                    f"Wrong type, expected int got `{type(y_pred)}`"
                )
                assert 0 <= y_pred < n_classes, (
                    f"Wrong class, expected 0 <= y_pred < {n_classes} but got {y_pred}."
                )

            else:
                assert y_pred is None and y_proba is None

        assert did_predict, f"Over {length} steps nothing was predicted."

    def test_save_then_load(self, case: Case, tmp_path: Path):
        """Check that a learner saved and then loaded behaves the same as the original."""
        if case.skip_test_save_then_load is not None:
            pytest.skip(case.skip_test_save_then_load)

        length = 50
        n_classes = 5

        stream = RandomTreeGenerator(num_classes=n_classes)
        learner = case.new_learner(stream.get_schema())

        # Initialize the learner with state
        for instance in islice(stream, length):
            learner.train(instance)

        # Save-then-load
        model_path = tmp_path / "model.pkl"
        with open(model_path, "wb") as f:
            save_model(learner, f)
        with open(model_path, "rb") as f:
            loaded: Classifier = load_model(f)  # type: ignore

        # Check constructor parameters match
        assert learner.get_params() == loaded.get_params()

        # Check that the original and the loaded clone match, step by step. The
        # stream keeps state, so this continues where the training loop stopped.
        for instance in islice(stream, length):
            y_pred_learner = learner.predict(instance)
            y_pred_loaded = loaded.predict(instance)
            assert y_pred_learner == y_pred_loaded, (
                "The saved-then-loaded clone differs from the reference prediction, "
                f"expected {y_pred_learner} but got {y_pred_loaded}."
            )
            y_prob_learner = learner.predict_proba(instance)
            y_prob_loaded = loaded.predict_proba(instance)
            assert _probas_close(y_prob_learner, y_prob_loaded), (
                "The saved-then-loaded clone differs from the reference proba, "
                f"expected {y_prob_learner} but got {y_prob_loaded}."
            )

            learner.train(instance)
            loaded.train(instance)

    def test_determinism(self, case: Case):
        """Check that learner is deterministic."""
        length = case.length_test_determinism
        n_classes = 10
        if case.skip_test_determinism is not None:
            pytest.skip(case.skip_test_determinism)

        preds_differ = False

        stream = RandomTreeGenerator(num_classes=n_classes)
        l1_a = case.new_learner(stream.get_schema(), random_seed=1)
        l1_b = case.new_learner(stream.get_schema(), random_seed=1)
        l2 = case.new_learner(stream.get_schema(), random_seed=2)

        for i, instance in enumerate(islice(stream, length)):
            # Get predictions
            l1a_y_pred = l1_a.predict(instance)
            l1b_y_pred = l1_b.predict(instance)
            l2_y_pred = l2.predict(instance)
            l1a_y_proba = l1_a.predict_proba(instance)
            l1b_y_proba = l1_b.predict_proba(instance)
            l2_y_proba = l2.predict_proba(instance)

            assert l1a_y_pred == l1b_y_pred, (
                "Expected learners with the same seed to predict the same, "
                f"but got {l1a_y_pred} and {l1b_y_pred} after {i} instances."
            )
            assert _probas_close(l1a_y_proba, l1b_y_proba), (
                "Expected learners with the same seed to have the same predicted "
                f"probas, but got {l1a_y_proba} and {l1b_y_proba} after {i} instances."
            )

            if case.always_deterministic:
                # Declared deterministic: seed 1 and seed 2 must match
                # exactly, not just reproducible (same seed gives the
                # same result).
                assert l1a_y_pred == l2_y_pred, (
                    f"{case.id} is declared always_deterministic, but seeds 1 "
                    f"and 2 predicted differently ({l1a_y_pred} vs {l2_y_pred}) "
                    f"after {i} instances."
                )
                assert _probas_close(l1a_y_proba, l2_y_proba), (
                    f"{case.id} is declared always_deterministic, but seeds 1 "
                    f"and 2 produced different probabilities "
                    f"({l1a_y_proba} vs {l2_y_proba}) after {i} instances."
                )
            else:
                # Only compare non-abstaining outputs, so a learner that abstains
                # earlier than another does not count as a difference.
                if (
                    l1a_y_pred is not None
                    and l2_y_pred is not None
                    and l1a_y_pred != l2_y_pred
                ):
                    preds_differ = True
                if (
                    l1a_y_proba is not None
                    and l2_y_proba is not None
                    and not _probas_close(l1a_y_proba, l2_y_proba)
                ):
                    preds_differ = True

            l1_a.train(instance)
            l1_b.train(instance)
            l2.train(instance)

        if not case.always_deterministic:
            assert preds_differ, (
                "Expected learners with different seeds to produce at least one"
                " different prediction."
            )
