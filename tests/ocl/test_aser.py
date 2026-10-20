"""ASER scoring, memory updates and classifier API checks."""

import copy
from itertools import permutations
from math import factorial

import numpy as np
import pytest

pytestmark = pytest.markskip("torch")

import torch
from torch import nn

from capymoa.core import Instance, LabeledInstance
from capymoa.ocl.strategy import ASER
from capymoa.ocl.strategy._aser import _knn_shapley
from capymoa.stream import Schema


def _tiny_aser(**kwargs) -> ASER:
    schema = Schema.from_custom(
        ["x0", "x1", "class"], "class", {"class": ["a", "b", "c"]}
    )
    model = kwargs.pop("model", None)
    if model is None:
        model = nn.Linear(2, 3)
    optimiser = kwargs.pop("optimiser", torch.optim.SGD(model.parameters(), lr=0.03))
    options = {"buffer_capacity": 8, "samples_per_class": 1, "knn_k": 3, "seed": 4}
    options.update(kwargs)
    return ASER(schema, model, optimiser, **options)


@pytest.mark.parametrize("k", [1, 2, 4])
def test_aser_shapley_matches_exhaustive_utility(k):
    """Check the recurrence against all coalitions, independently of its formula."""
    candidates = torch.tensor([[0.1], [0.4], [0.8], [1.5]], dtype=torch.float64)
    labels = torch.tensor([0, 1, 0, 0])
    exact = torch.zeros(4, dtype=torch.float64)
    for permutation in permutations(range(4)):
        coalition, previous = [], 0.0
        for index in permutation:
            coalition.append(index)
            nearest = sorted(coalition)[:k]  # Distances increase with candidate index.
            utility = sum(labels[i] == 0 for i in nearest) / k
            exact[index] += utility - previous
            previous = utility
    exact /= factorial(4)
    actual = _knn_shapley(
        torch.zeros(1, 1, dtype=torch.float64), torch.tensor([0]), candidates, labels, k
    )
    torch.testing.assert_close(actual[0], exact)


def test_aser_shapley_keeps_reference_small_candidate_behavior():
    actual = _knn_shapley(
        torch.zeros(1, 1),
        torch.tensor([0]),
        torch.tensor([[1.0], [2.0]]),
        torch.tensor([0, 0]),
        5,
    )
    torch.testing.assert_close(actual, torch.tensor([[0.5, 0.5]]))


@pytest.mark.parametrize(
    "mode, expected", [("asv", [0, 2]), ("asvm", [2, 0]), ("neg_sv", [2, 0])]
)
def test_aser_retrieval_modes(mode, expected, monkeypatch):
    learner = _tiny_aser(score_type=mode, replay_batch_size=2)
    x = torch.arange(16, dtype=torch.float32).reshape(8, 2)
    y = torch.tensor([0] * 4 + [1] * 4)
    learner._update_memory(x, y)
    learner._seen = 9
    candidates, evaluation = torch.tensor([0, 2, 4]), torch.tensor([1, 5])
    monkeypatch.setattr(
        learner,
        "_balanced_indices",
        lambda excluded=None: candidates if excluded is None else evaluation,
    )
    adv = torch.tensor([[-0.8, 0.2, -0.2], [0.6, 0.4, -0.1]])
    coop = torch.tensor([[0.4, 0.1, 0.5], [0.2, 0.3, 0.4]])
    calls = []

    def shapley(ex, ey, cx, cy):
        calls.append(ex)
        return adv if len(calls) == 1 else coop

    monkeypatch.setattr(learner, "_shapley", shapley)
    replay = learner._retrieve(torch.eye(2), torch.tensor([2, 2]))
    torch.testing.assert_close(replay["x"], x[candidates[expected]])
    assert len(calls) == (1 if mode == "neg_sv" else 2)


def test_aser_shapley_replacement_and_minority_class(monkeypatch):
    learner = _tiny_aser(samples_per_class=2)
    x = torch.arange(16, dtype=torch.float32).reshape(8, 2)
    y = torch.tensor([0] * 4 + [1] * 4)
    learner._update_memory(x, y)
    evaluation, candidates = torch.tensor([0, 2, 4, 6]), torch.tensor([1, 3, 5, 7])
    monkeypatch.setattr(learner, "_balanced_indices", lambda excluded=None: evaluation)
    monkeypatch.setattr(
        learner, "_random_indices", lambda count, excluded=None: candidates
    )
    new_x, new_y = torch.tensor([[100.0, 101.0], [200.0, 201.0]]), torch.tensor([2, 0])

    def shapley(ex, ey, cx, cy):
        # A previously absent class must be added to the evaluation set.
        torch.testing.assert_close(ex, torch.cat((x[evaluation], new_x[:1])))
        torch.testing.assert_close(cx, torch.cat((x[candidates], new_x)))
        return torch.tensor([[0.4, 0.3, 0.2, 0.1, 1.0, -1.0]]).expand(len(ex), -1)

    monkeypatch.setattr(learner, "_shapley", shapley)
    learner._update_memory(new_x, new_y)
    expected = x.clone()
    expected[7] = new_x[0]
    torch.testing.assert_close(learner._buffer.array()["x"], expected)
    assert learner._class_indices == {0: {0, 1, 2, 3}, 1: {4, 5, 6}, 2: {7}}
    assert learner._buffer.count == 8 and learner._seen == 10


def test_aser_warmup_sampling_and_small_memory(monkeypatch):
    learner = _tiny_aser(replay_batch_size=20)
    x, y = (
        torch.arange(16, dtype=torch.float32).reshape(8, 2),
        torch.tensor([0] * 4 + [1] * 4),
    )
    learner._update_memory(x, y)

    def unexpected(*args):
        raise AssertionError("ASER retrieval must wait until seen > capacity")

    monkeypatch.setattr(learner, "_shapley", unexpected)
    replay = learner._retrieve(torch.eye(2), torch.tensor([0, 1]))
    assert len(replay["y"]) == 8
    assert len(replay["x"].unique(dim=0)) == 8
    selected = learner._balanced_indices()
    assert len(selected) == 2
    assert len(learner._balanced_indices(selected)) == 2
    small = _tiny_aser(buffer_capacity=1)
    small.batch_train(torch.eye(2), torch.tensor([0, 1]))
    small.batch_train(torch.eye(2), torch.tensor([1, 2]))
    assert small._seen == 4 and small._buffer.count == 1
    assert np.isfinite(small.last_loss)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_aser_single_training_pass_and_prediction(dtype):
    model = nn.Sequential(
        nn.Linear(2, 4), nn.BatchNorm1d(4), nn.ReLU(), nn.Linear(4, 3)
    ).to(dtype)
    learner = _tiny_aser(model=model)
    x, y = torch.eye(2).repeat(2, 1), torch.tensor([0, 1, 0, 1])
    learner.batch_train(x, y)
    assert int(model[1].num_batches_tracked) == 1
    learner.batch_train(x, y)
    assert int(model[1].num_batches_tracked) == 2
    model[1].eval()
    modes = [m.training for m in model.modules()]
    state = {name: value.clone() for name, value in model.state_dict().items()}
    learner._features(x.to(dtype))
    probabilities = learner.batch_predict_proba(x)
    torch.testing.assert_close(probabilities.sum(1), torch.ones(4, dtype=dtype))
    assert modes == [m.training for m in model.modules()]
    for name, value in model.state_dict().items():
        torch.testing.assert_close(value, state[name])
    instance = Instance.from_array(learner.schema, x[0].numpy())
    np.testing.assert_allclose(
        learner.predict_proba(instance), probabilities[0].numpy(), rtol=1e-5
    )
    # With empty memory, a batch-normalized encoder still cannot train a singleton.
    single = _tiny_aser(model=nn.Linear(2, 3).to(dtype))
    single.train(LabeledInstance.from_array(single.schema, x[0].numpy(), 0))
    assert single._seen == 1


@pytest.mark.parametrize("substeps", [1, 2])
def test_aser_combined_update_with_singleton_and_replay(substeps, monkeypatch):
    """A singleton plus replay must match one combined SGD update, including state."""
    torch.manual_seed(7)
    model = nn.Sequential(
        nn.Linear(2, 4), nn.BatchNorm1d(4), nn.ReLU(), nn.Dropout(0.2), nn.Linear(4, 3)
    )
    expected_model = copy.deepcopy(model)
    optimiser = torch.optim.SGD(model.parameters(), lr=0.03, momentum=0.9)
    expected_optimiser = torch.optim.SGD(
        expected_model.parameters(), lr=0.03, momentum=0.9
    )
    learner = _tiny_aser(model=model, optimiser=optimiser, substeps=substeps)
    replay_x = torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    replay_y = torch.tensor([0, 1, 0])
    learner._update_memory(replay_x, replay_y)
    x, y = torch.tensor([[0.2, 0.3]]), torch.tensor([2])

    def retrieve(current_x, current_y):
        # Current data must not enter memory between substeps.
        assert learner._buffer.count == len(replay_y)
        return {"x": replay_x, "y": replay_y}

    monkeypatch.setattr(learner, "_retrieve", retrieve)
    combined_x = torch.cat((replay_x, x))
    combined_y = torch.cat((replay_y, y))
    rng = torch.random.get_rng_state().clone()
    expected_losses = []
    for _ in range(substeps):
        expected_optimiser.zero_grad()
        loss = nn.functional.cross_entropy(expected_model(combined_x), combined_y)
        loss.backward()
        expected_optimiser.step()
        expected_losses.append(loss.detach())
    expected_rng = torch.random.get_rng_state().clone()

    # Replay selection is fixed so dropout consumes exactly the same random draws.
    torch.random.set_rng_state(rng)
    learner.batch_train(x, y)
    torch.testing.assert_close(torch.random.get_rng_state(), expected_rng)
    for name, value in model.state_dict().items():
        torch.testing.assert_close(value, expected_model.state_dict()[name])
    for actual, expected in zip(model.parameters(), expected_model.parameters()):
        torch.testing.assert_close(
            optimiser.state[actual]["momentum_buffer"],
            expected_optimiser.state[expected]["momentum_buffer"],
        )
    assert learner.last_loss == pytest.approx(
        torch.stack(expected_losses).mean().item()
    )
    assert learner._buffer.count == learner._seen == len(combined_y)


def test_aser_reproducibility_and_private_cache():
    results = []
    for _ in range(2):
        model = nn.Linear(2, 3)
        with torch.no_grad():
            model.weight.fill_(0.2)
            model.bias.zero_()
        learner = _tiny_aser(model=model)
        x, y = (
            torch.arange(16, dtype=torch.float32).reshape(8, 2) / 16,
            torch.tensor([0, 1] * 4),
        )
        for offset in range(4):
            learner.batch_train(x + offset, (y + offset) % 3)
        results.append(learner)
    for key, value in results[0]._model.state_dict().items():
        torch.testing.assert_close(value, results[1]._model.state_dict()[key])
    torch.testing.assert_close(
        results[0]._buffer.array()["x"], results[1]._buffer.array()["x"]
    )
    assert results[0]._class_indices == results[1]._class_indices
    assert results[0]._class_indices is not results[1]._class_indices
    # Sampling itself must not consume the caller's global random stream.
    state = torch.random.get_rng_state().clone()
    results[0]._random_indices(2)
    torch.testing.assert_close(torch.random.get_rng_state(), state)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"buffer_capacity": 0},
        {"replay_batch_size": 0},
        {"samples_per_class": 0},
        {"knn_k": 0},
        {"substeps": 0},
        {"feature_batch_size": 0},
        {"score_type": "invalid"},
    ],
)
def test_aser_invalid_configuration(kwargs):
    with pytest.raises(ValueError):
        _tiny_aser(**kwargs)
