"""Tests for the ``module`` interface of torch-backed OCL strategies."""

import pytest
import torch
from torch import nn

from capymoa.core.torch.ann import Perceptron
from capymoa.ocl.datasets import TinySplitMNIST
from capymoa.ocl.strategy import DER, DERPP, EWC, LWF, MAS, SCR, SI, RWalk

pytestmark = pytest.mark.torch


class _ScrModel(nn.Module):
    def __init__(self, n_in: int):
        super().__init__()
        self.feature_extractor = nn.Sequential(
            nn.Flatten(), nn.Linear(n_in, 16), nn.ReLU()
        )
        self.projection = nn.Linear(16, 8)


def _make(cls, schema):
    if cls is SCR:
        model = _ScrModel(schema.get_num_attributes())
        opt = torch.optim.SGD(model.parameters(), lr=0.1)
        return SCR(
            schema, model, opt, nn.Identity(), buffer_capacity=64, replay_batch_size=8
        )
    model = Perceptron(schema)
    opt = torch.optim.SGD(model.parameters(), lr=0.1)
    if cls is EWC:
        return EWC(schema, model, opt, lambda_=1.0)
    if cls is MAS:
        return MAS(schema, model, opt, lambda_=1.0)
    if cls in (RWalk, SI):
        return cls(schema, model, opt, lambda_=1.0)
    return cls(schema, model, opt)


CLASSES = [EWC, LWF, MAS, RWalk, SI, DER, DERPP, SCR]
NN_MODULE_ONLY = [EWC, LWF, MAS, RWalk, SI]


@pytest.fixture(scope="module")
def schema():
    return TinySplitMNIST().schema


@pytest.mark.parametrize("cls", CLASSES)
def test_module_contains_model(cls, schema):
    learner = _make(cls, schema)
    assert isinstance(learner.module, nn.Module)
    assert isinstance(learner.module.model, nn.Module)
    assert learner.module is learner._module


@pytest.mark.parametrize("cls", CLASSES)
def test_module_is_read_only(cls, schema):
    learner = _make(cls, schema)
    with pytest.raises(AttributeError):
        learner.module = nn.Linear(1, 1)  # type: ignore


ROUND_TRIP = [c for c in CLASSES if c is not SCR]


@pytest.mark.parametrize("cls", ROUND_TRIP)
def test_state_dict_round_trip(cls, schema):
    torch.manual_seed(0)
    n = schema.get_num_attributes()
    x = torch.rand(16, n)
    y = torch.randint(0, schema.get_num_classes(), (16,))

    trained = _make(cls, schema)
    for _ in range(3):
        trained.batch_train(x, y)
    expected = trained.batch_predict_proba(x)

    fresh = _make(cls, schema)
    assert not torch.allclose(fresh.batch_predict_proba(x), expected)
    fresh.module.load_state_dict(trained.module.state_dict())
    torch.testing.assert_close(fresh.batch_predict_proba(x), expected)


def test_scr_state_dict_round_trip(schema):
    """SCR weights and replay tensors restore.

    Predictions do not, because the buffer count is a plain int outside the
    state dict, so a loaded SCR sees an empty buffer.
    """
    n = schema.get_num_attributes()
    x = torch.rand(16, n)
    y = torch.randint(0, schema.get_num_classes(), (16,))
    trained, fresh = _make(SCR, schema), _make(SCR, schema)
    trained.batch_train(x, y)
    fresh.module.load_state_dict(trained.module.state_dict())
    expected = trained.module.state_dict()
    for k, v in fresh.module.state_dict().items():
        torch.testing.assert_close(v, expected[k])


@pytest.mark.parametrize("cls", NN_MODULE_ONLY)
def test_not_an_nn_module(cls, schema):
    assert nn.Module not in cls.__mro__
    learner = _make(cls, schema)
    assert not isinstance(learner, nn.Module)
    for name in ("eval", "cuda", "half", "forward", "apply", "parameters"):
        assert not hasattr(learner, name)


@pytest.mark.parametrize("cls", CLASSES)
def test_no_state_dict_on_strategy(cls, schema):
    learner = _make(cls, schema)
    assert not hasattr(learner, "state_dict")
    assert not hasattr(learner, "load_state_dict")
