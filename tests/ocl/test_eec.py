"""Paper equations, memory integration and task-event checks for EEC."""

import copy
from math import exp

import numpy as np
import pytest

pytestmark = pytest.markskip("torch")

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from capymoa.core import Instance
from capymoa.ocl import evaluate_ocl
from capymoa.ocl.evaluation.events import TestBegin, TrainTaskBegin, TrainTaskEnd
from capymoa.ocl.events import Dispatcher
from capymoa.ocl.strategy import EEC
from capymoa.ocl.strategy._eec import _Autoencoder, _Concept, _merge, _quotas
from capymoa.stream import Schema


class TinyAutoencoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(nn.Flatten(), nn.Linear(64, 4))
        self.decoder = nn.Sequential(nn.Linear(4, 64), nn.Unflatten(1, (1, 8, 8)))


@pytest.fixture
def schema():
    schema = Schema.from_custom(
        features=[*[f"x{i}" for i in range(64)], "class"],
        target="class",
        categories={"class": ["zero", "one"]},
    )
    schema.shape = (1, 8, 8)
    return schema


def learner(schema, **kwargs):
    torch.manual_seed(7)
    model = nn.Sequential(nn.Linear(64, 8), nn.Tanh(), nn.Linear(8, 2))
    defaults = {
        "epochs": 2,
        "subsequent_epochs": 2,
        "autoencoder_epochs": 1,
        "batch_size": 4,
        "feature_extractor": model[0],
        "autoencoder_factory": TinyAutoencoder,
    }
    defaults.update(kwargs)
    return EEC(schema, model, **defaults)


def test_merge_matches_direct_sample_covariance():
    x = torch.tensor([[0.0, 1.0], [2.0, 3.0], [4.0, 9.0], [8.0, 2.0]])
    concepts = [_Concept(row, 0, 0) for row in x]
    result = _merge(_merge(concepts[0], concepts[1]), _merge(concepts[2], concepts[3]))
    torch.testing.assert_close(result.mean, x.mean(0))
    torch.testing.assert_close(result.variance, torch.var(x, dim=0, unbiased=True))
    assert result.count == 4 and result.units == 2
    with pytest.raises(ValueError, match="same task and class"):
        _merge(result, _Concept(x[0], 1, 0))


def test_proportional_quotas_and_minimum_representation():
    groups = [
        [_Concept(torch.tensor([float(i)]), label, 0) for i in range(n)]
        for label, n in enumerate((6, 3))
    ]
    assert _quotas(groups, 6) == [4, 2]
    assert _quotas(groups, 4) == [2, 2]
    with pytest.raises(ValueError, match="capacity"):
        _quotas(groups, 3)


def test_integration_preserves_counts_and_latest_episodes(schema):
    strategy = learner(schema, capacity=8)
    strategy._concepts = [_Concept(torch.tensor([float(i)]), 0, 0) for i in range(6)]
    latest = [_Concept(torch.tensor([float(i)]), 1, 1) for i in range(4)]
    integrated = strategy._integrate(latest)
    assert sum(c.units for c in integrated) <= 8
    assert sum(c.count for c in integrated if c.task == 0) == 6
    assert integrated[-4:] == latest
    assert all(c.variance is None for c in latest)
    # Merging a pair must count as two vectors, not one concept object.
    assert any(c.count > 1 and c.units == 2 for c in integrated)
    assert len(strategy._concepts) == 6  # Pure planning leaves old memory intact.


def test_impossible_budget_fails_without_dropping_classes(schema):
    strategy = learner(schema, capacity=5)
    strategy._concepts = [_Concept(torch.tensor([float(i)]), 0, 0) for i in range(4)]
    with pytest.raises(ValueError, match="capacity"):
        strategy._integrate([_Concept(torch.tensor([0.0]), 1, 1) for _ in range(4)])
    assert len(strategy._concepts) == 4


class SignClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(1.0))

    def forward(self, x):
        score = x.flatten(1).mean(1) * self.scale
        return torch.stack((-score, score), dim=1)


class IdentityAutoencoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Identity()
        self.decoder = nn.Identity()


def test_pseudo_images_reject_wrong_labels_without_fallback(schema):
    strategy = EEC(schema, SignClassifier(), content_weight=0, filter_attempts=2)
    strategy._autoencoders[0] = IdentityAutoencoder()
    mean = torch.ones(1, 8, 8)
    concept = _Concept(mean, 1, 0, 3, torch.zeros_like(mean))
    images = strategy._pseudo_images(concept)
    assert images.shape == (3, 1, 8, 8)
    assert (strategy.batch_predict(images) == 1).all()
    concept.label = 0
    with pytest.raises(RuntimeError, match="No mismatched image"):
        strategy._pseudo_images(concept)


def test_replay_multiplicity_and_fixed_reconstruction_depth(schema):
    strategy = EEC(schema, SignClassifier(), content_weight=0)
    strategy._autoencoders[0] = IdentityAutoencoder()
    strategy._original_accuracy[0] = 1.0
    positive = torch.ones(1, 8, 8)
    negative = -positive
    strategy._concepts = [
        _Concept(positive, 1, 0),
        _Concept(negative, 1, 0),
        _Concept(positive, 1, 0, 3, torch.zeros_like(positive)),
    ]
    replay = strategy._replay()
    assert len(replay[0][0]) == 2 and replay[0][2] == pytest.approx(exp(-0.5))
    assert len(replay[1][0]) == 3 and replay[1][2] == 1.0
    # Unrelated later decoders must not turn reconstruction depth into task age.
    strategy._autoencoders[99] = IdentityAutoencoder()
    assert strategy._replay()[0][2] == replay[0][2]
    strategy._original_accuracy[0] = 0.0
    assert strategy._replay()[0][2] == 1.0


def test_classifier_uses_sum_of_source_means(schema):
    model = nn.Linear(64, 2, bias=False)
    nn.init.zeros_(model.weight)
    strategy = EEC(
        schema,
        model,
        torch.optim.SGD(model.parameters(), lr=0.1),
        content_weight=0,
        epochs=1,
        batch_size=4,
    )
    source = torch.ones(1, 1, 8, 8)
    replay = source * 2
    strategy._replay = lambda: [(replay, torch.tensor([1]), 0.25)]
    dispatcher = Dispatcher()
    strategy.attach_with(dispatcher)
    dispatcher.notify(TrainTaskBegin(0, 0))
    strategy.batch_train(source.flatten(1), torch.tensor([0]))
    before = copy.deepcopy(model)
    expected_loss = nn.functional.cross_entropy(
        before(source.flatten(1)), torch.tensor([0])
    )
    expected_loss += 0.25 * nn.functional.cross_entropy(
        before(replay.flatten(1)), torch.tensor([1])
    )
    expected_loss.backward()
    expected = before.weight.detach() - 0.1 * before.weight.grad
    dispatcher.notify(TestBegin())
    torch.testing.assert_close(model.weight, expected)


def test_content_loss_updates_autoencoder_but_freezes_classifier(schema):
    strategy = learner(schema, content_weight=1.0)
    strategy.model.train()
    strategy.model[0].eval()  # Preserve mixed submodule modes too.
    flags = [p.requires_grad for p in strategy.model.parameters()]
    before = {
        name: value.clone() for name, value in strategy.model.state_dict().items()
    }
    fresh = TinyAutoencoder()
    initial = {name: value.clone() for name, value in fresh.state_dict().items()}
    strategy._autoencoder_factory = lambda: fresh
    result = strategy._train_autoencoder(torch.randn(4, 1, 8, 8))
    assert any(
        not torch.equal(initial[name], value)
        for name, value in result.state_dict().items()
    )
    assert all(
        torch.equal(before[name], value)
        for name, value in strategy.model.state_dict().items()
    )
    assert flags == [p.requires_grad for p in strategy.model.parameters()]
    assert strategy.model.training and not strategy.model[0].training
    assert not any(p.requires_grad for p in result.parameters())


def test_task_events_fit_before_evaluation_and_release_final_cache(schema):
    strategy = learner(schema)
    dispatcher = Dispatcher()
    strategy.attach_with(dispatcher)
    with pytest.raises(RuntimeError, match="events"):
        strategy.batch_train(torch.ones(2, 64), torch.zeros(2, dtype=torch.long))
    dispatcher.notify(TrainTaskBegin(0, 0))
    x = torch.ones(4, 64)
    y = torch.zeros(4, dtype=torch.long)
    original = copy.deepcopy(strategy.model.state_dict())
    strategy.batch_train(x, y)
    x.zero_()  # Cache owns its data.
    assert torch.cat(strategy._task_x).sum() == 256
    assert all(
        torch.equal(original[k], v) for k, v in strategy.model.state_dict().items()
    )
    dispatcher.notify(TestBegin())
    assert strategy._fitted
    fitted = copy.deepcopy(strategy.model.state_dict())
    dispatcher.notify(TestBegin())
    assert all(
        torch.equal(fitted[k], v) for k, v in strategy.model.state_dict().items()
    )
    with pytest.raises(RuntimeError, match="continual_evaluations=1"):
        strategy.batch_train(x, y)
    dispatcher.notify(TrainTaskEnd(0, 4))
    assert not strategy._task_x and not strategy._task_y
    assert strategy.memory_units == 4 and set(strategy._autoencoders) == {0}
    dispatcher.notify(TrainTaskEnd(0, 4))
    assert strategy.memory_units == 4
    with pytest.raises(ValueError, match="new task ID"):
        dispatcher.notify(TrainTaskBegin(0, 4))


def test_default_autoencoder_matches_paper_downsampling(schema):
    ae = _Autoencoder(1)
    x = torch.randn(2, 1, 32, 32)
    assert ae.encoder(x).shape == (2, 16, 4, 4)
    assert ae(x).shape == x.shape
    strategy = learner(schema, autoencoder_factory=None)
    trained = strategy._train_autoencoder(torch.ones(1, 1, 8, 8))
    assert trained(torch.ones(1, 1, 8, 8)).shape == (1, 1, 8, 8)


def test_memory_accounts_for_models_and_task_cache(schema):
    strategy = learner(schema)
    strategy._begin_task(TrainTaskBegin(0, 0))
    strategy.batch_train(torch.ones(4, 64), torch.zeros(4, dtype=torch.long))
    before = strategy.memory_breakdown_bytes()
    assert before["active_task"] == 4 * (64 * 4 + 8)
    strategy._end_task(TrainTaskEnd(0, 4))
    after = strategy.memory_breakdown_bytes()
    assert after["active_task"] == 0 and after["autoencoders"] > 0
    assert after["latent"] == 4 * 4 * 4  # Four four-dimensional float32 codes.
    assert after["total"] == sum(v for k, v in after.items() if k != "total")


def test_prediction_preserves_modes_and_supports_instances(schema):
    strategy = learner(schema)
    strategy.model.train()
    strategy.model[0].eval()
    x = torch.ones(2, 64)
    probabilities = strategy.batch_predict_proba(x)
    torch.testing.assert_close(probabilities.sum(1), torch.ones(2))
    assert strategy.model.training and not strategy.model[0].training
    instance = Instance.from_array(schema, np.ones(64))
    assert strategy.predict_proba(instance).shape == (2,)


def test_autoencoder_creation_preserves_global_rng(schema):
    strategy = learner(schema)
    state = torch.random.get_rng_state().clone()
    strategy._new_autoencoder()
    assert torch.equal(state, torch.random.get_rng_state())


@pytest.mark.parametrize(
    "kwargs",
    [
        {"capacity": 0},
        {"epochs": 0},
        {"batch_size": True},
        {"filter_attempts": 0},
        {"content_weight": 1.1},
        {"autoencoder_lr": float("nan")},
        {"autoencoder_weight_decay": -1.0},
    ],
)
def test_invalid_configuration(schema, kwargs):
    with pytest.raises(ValueError):
        learner(schema, **kwargs)


def test_evaluator_integration_for_two_tasks(schema):
    strategy = learner(schema)
    loaders = [
        DataLoader(
            TensorDataset(
                torch.full((4, 1, 8, 8), float(label)),
                torch.full((4,), label, dtype=torch.long),
            ),
            batch_size=2,
        )
        for label in range(2)
    ]
    results = evaluate_ocl(strategy, loaders, loaders)
    assert results["accuracy_matrix"].shape == (2, 2)
    assert strategy._active_task is None and not strategy._task_x
    assert strategy.memory_units == 8 and len(strategy._autoencoders) == 2


class Scale(nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(1.0))

    def forward(self, x):
        return x * self.scale


class ScaleAutoencoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Identity()
        self.decoder = Scale()


def test_compressed_memory_is_replayed_through_its_own_decoder(schema):
    strategy = EEC(
        schema,
        SignClassifier(),
        content_weight=0,
        autoencoder_factory=ScaleAutoencoder,
        capacity=8,
        epochs=1,
        subsequent_epochs=1,
        autoencoder_epochs=1,
        batch_size=4,
    )
    loaders = [
        DataLoader(
            TensorDataset(
                torch.full((4, 1, 8, 8), sign),
                torch.full((4,), label, dtype=torch.long),
            ),
            batch_size=2,
        )
        for sign, label in [(-1.0, 0), (1.0, 1), (-1.0, 0)]
    ]
    result = evaluate_ocl(strategy, loaders, loaders)
    assert result["accuracy_final"] == 1.0
    assert strategy.memory_units == 8
    assert sum(c.count for c in strategy._concepts) == 12
    assert len(strategy._autoencoders) == 3


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_cuda_instance_prediction_and_task_finalisation(schema):
    strategy = learner(schema, device="cuda")
    strategy._begin_task(TrainTaskBegin(0, 0))
    strategy.batch_train(
        torch.ones(4, 64, device="cuda"),
        torch.zeros(4, dtype=torch.long, device="cuda"),
    )
    strategy._end_task(TrainTaskEnd(0, 4))
    assert strategy.predict_proba(Instance.from_array(schema, np.ones(64))).shape == (
        2,
    )
