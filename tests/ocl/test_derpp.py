"""DER++ objective, DER compatibility and shared-core integration checks."""

import copy
from unittest.mock import patch

import pytest

pytestmark = pytest.markskip("torch")

import torch
import torch.nn.functional as F
from torch import nn

from capymoa.core import Instance, LabeledInstance
from capymoa.ocl.strategy import DER, DERPP
from capymoa.ocl.util._replay import ReservoirSampler
from capymoa.stream import Schema


def _schema(features=2):
    return Schema.from_custom(
        [f"x{i}" for i in range(features)] + ["class"],
        "class",
        {"class": ["0", "1"]},
    )


def _learner(model=None, schema=None, **kwargs):
    schema = schema if schema is not None else _schema()
    if model is None:
        torch.manual_seed(0)
        model = nn.Linear(schema.get_num_attributes(), 2)
    return DERPP(
        schema,
        model,
        torch.optim.SGD(model.parameters(), lr=0.1),
        buffer_capacity=kwargs.pop("buffer_capacity", 8),
        **kwargs,
    )


def _batch():
    return torch.eye(2), torch.tensor([0, 1])


@pytest.mark.parametrize("substeps", [1, 3])
def test_beta_zero_matches_der_parameters_buffer_and_rng(substeps):
    learner = _learner(beta=0, seed=7, substeps=substeps, buffer_capacity=3)
    reference_model = copy.deepcopy(learner._model)
    reference = DER(
        learner.schema,
        reference_model,
        torch.optim.SGD(reference_model.parameters(), lr=0.1),
        buffer_capacity=3,
        seed=7,
        substeps=substeps,
    )
    x, y = _batch()
    for offset in range(5):
        learner.batch_train(x + offset, y)
        reference.batch_train(x + offset, y)
    for actual, expected in zip(
        learner._model.parameters(), reference._model.parameters()
    ):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    for key in ("x", "y", "z"):
        torch.testing.assert_close(
            learner._buffer.array()[key], reference._buffer.array()[key]
        )
    assert torch.equal(
        learner._buffer._rng.get_state(), reference._buffer._rng.get_state()
    )


def test_combined_objective_matches_one_manual_optimizer_step():
    learner = _learner(alpha=0.2, beta=0.7, seed=4)
    memory_x = torch.tensor([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]])
    memory_y = torch.tensor([0, 1, 1])
    memory_z = torch.tensor([[2.0, -1.0], [-0.5, 1.0], [-1.0, 0.5]])
    learner._buffer.update(x=memory_x, y=memory_y, z=memory_z)
    rng_state = learner._buffer._rng.get_state()
    dark = learner._buffer.sample(2)
    labels = learner._buffer.sample(2)
    learner._buffer._rng.set_state(rng_state)
    reference = copy.deepcopy(learner._model)
    optimizer = torch.optim.SGD(reference.parameters(), lr=0.1)
    x = torch.tensor([[1.0, 1.0], [-1.0, 1.0]])
    y = torch.tensor([1, 0])
    expected = (
        F.cross_entropy(reference(x), y)
        + 0.2 * F.mse_loss(reference(dark["x"]), dark["z"])
        + 0.7 * F.cross_entropy(reference(labels["x"]), labels["y"])
    )
    optimizer.zero_grad()
    expected.backward()
    optimizer.step()
    with patch.object(
        learner._optimiser, "step", wraps=learner._optimiser.step
    ) as step:
        learner.batch_train(x, y)
    assert step.call_count == 1
    for actual, expected in zip(learner._model.parameters(), reference.parameters()):
        torch.testing.assert_close(actual, expected)


def test_replay_draws_are_independent_and_precede_current_batch_insertion():
    learner = _learner(buffer_capacity=1)
    x, y = _batch()
    with patch.object(
        learner._buffer, "sample", wraps=learner._buffer.sample
    ) as sample:
        learner.batch_train(x, y)
    assert sample.call_count == 0
    historical_x = learner._buffer.array()["x"].clone()
    draws = []
    original_sample = learner._buffer.sample

    def sample(n):
        # Current examples must not enter memory until both draws complete.
        torch.testing.assert_close(learner._buffer.array()["x"], historical_x)
        result = original_sample(n)
        draws.append(result)
        return result

    with patch.object(learner._buffer, "sample", side_effect=sample):
        learner.batch_train(x + 10, y)
    assert len(draws) == 2
    assert draws[0] is not draws[1]
    for draw in draws:
        assert draw["x"].shape == (2, 2)  # DER samples with replacement.
        torch.testing.assert_close(draw["x"], historical_x.expand(2, -1))


def test_named_reservoir_preserves_pre_update_logits_and_alignment():
    learner = _learner(buffer_capacity=3)
    assert isinstance(learner, DER)
    assert isinstance(learner._buffer, ReservoirSampler)
    reference = copy.deepcopy(learner._model)
    x = torch.arange(20, dtype=torch.float32).reshape(10, 2)
    y = torch.arange(10) % 2
    learner.batch_train(x, y)
    memory = learner._buffer.array()
    assert set(memory) == {"x", "y", "z"}
    assert learner._buffer.count == 3
    assert memory["x"].shape == (3, 2)  # No packed feature/logit storage.
    torch.testing.assert_close(memory["z"], reference(memory["x"]))
    torch.testing.assert_close(memory["y"], (memory["x"][:, 0].long() // 2) % 2)
    assert not memory["z"].requires_grad


def test_parameter_view_logits_are_snapshotted_before_optimization():
    class BiasOnly(nn.Module):
        def __init__(self):
            super().__init__()
            self.bias = nn.Parameter(torch.tensor([1.0, -1.0]))

        def forward(self, x):
            return self.bias.expand(x.shape[0], -1)

    learner = _learner(model=BiasOnly())
    before = learner._model.bias.detach().clone()
    learner.batch_train(torch.ones(2, 2), torch.tensor([1, 1]))
    torch.testing.assert_close(learner._buffer.array()["z"], before.expand(2, -1))
    assert not torch.allclose(learner._model.bias, before)


def test_substeps_use_existing_der_loop_and_insert_examples_only_once():
    learner = _learner(substeps=3)
    x, y = _batch()
    with (
        patch.object(learner._buffer, "update", wraps=learner._buffer.update) as update,
        patch.object(learner._optimiser, "step", wraps=learner._optimiser.step) as step,
        patch.object(learner._buffer, "sample", wraps=learner._buffer.sample) as sample,
    ):
        learner.batch_train(x, y)
    assert step.call_count == 3
    assert update.call_count == 1
    assert sample.call_count == 4  # Two draws on each of the last two substeps.
    assert learner._buffer.count == 2


def test_replay_rng_is_independent_of_unrelated_random_draws():
    first, second = _learner(seed=7), _learner(seed=7)
    x, y = _batch()
    for i in range(8):
        first.batch_train(x + i, y)
        _ = torch.rand(50)
        second.batch_train(x + i, y)
    for key in ("x", "y", "z"):
        torch.testing.assert_close(
            first._buffer.array()[key], second._buffer.array()[key]
        )
    torch.testing.assert_close(
        first.batch_predict_proba(x), second.batch_predict_proba(x)
    )


def test_schema_shape_augmentation_and_single_instance_interfaces():
    schema = _schema(features=4)
    schema.shape = (1, 2, 2)
    model = nn.Sequential(nn.Conv2d(1, 2, 1), nn.Flatten(), nn.Linear(8, 2))
    shapes = []

    def augment(x):
        shapes.append(tuple(x.shape))
        return x

    learner = _learner(schema=schema, model=model, augment=augment)
    x = torch.arange(8, dtype=torch.float32).reshape(2, 4)
    y = torch.tensor([0, 1])
    learner.batch_train(x, y)
    learner.batch_train(x, y)
    assert shapes == [(2, 1, 2, 2)] * 4
    probabilities = learner.batch_predict_proba(x)
    torch.testing.assert_close(probabilities.sum(1), torch.ones(2))
    instance = Instance.from_array(schema, x[0].numpy())
    assert learner.predict_proba(instance).dtype.name == "float64"
    learner.train(LabeledInstance.from_array(schema, x[0].numpy(), 0))
    assert learner._buffer.count == 5
    assert learner.get_params()["beta"] == 0.5


@pytest.mark.parametrize(
    "kwargs",
    [
        {"beta": -1},
        {"beta": float("nan")},
        {"alpha": float("inf")},
        {"buffer_capacity": 0},
        {"substeps": 0},
        {"substeps": 1.5},
    ],
)
def test_rejects_invalid_configuration(kwargs):
    with pytest.raises(ValueError):
        _learner(**kwargs)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_cuda_uses_der_device_placement_and_single_instance_prediction():
    learner = _learner(device="cuda")
    x, y = _batch()
    learner.batch_train(x, y)
    learner.batch_train(x, y)
    assert learner._buffer.device.type == "cuda"
    instance = Instance.from_array(learner.schema, x[0].numpy())
    assert learner.predict_proba(instance).shape == (2,)


def test_existing_ocl_loop_supports_derpp_without_task_callbacks():
    from torch.utils.data import DataLoader, TensorDataset

    from capymoa.ocl.evaluation import ocl_train_eval_loop

    learner = _learner()
    x, y = _batch()
    loader = DataLoader(TensorDataset(x, y), batch_size=1)
    results = ocl_train_eval_loop(
        learner, [loader, loader], [loader, loader], progress_bar=False
    )
    assert learner._buffer.count == 4
    assert 0 <= results.accuracy_final <= 1
