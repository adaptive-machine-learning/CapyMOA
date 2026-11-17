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
        self.feature_extractor = nn.Sequential(nn.Linear(n_in, 16), nn.ReLU())
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
    assert learner.module.model is learner._model
    assert learner.module is learner._module


@pytest.mark.parametrize("cls", CLASSES)
def test_module_is_read_only(cls, schema):
    learner = _make(cls, schema)
    with pytest.raises(AttributeError):
        learner.module = nn.Linear(1, 1)  # type: ignore


# SCR predicts from prototypes built on its replay buffer, which is not torch state.
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
    fresh.load_state_dict(trained.state_dict())
    torch.testing.assert_close(fresh.batch_predict_proba(x), expected)


def test_scr_state_dict_round_trip(schema):
    a, b = _make(SCR, schema), _make(SCR, schema)
    b.load_state_dict(a.state_dict())
    for pa, pb in zip(a.module.parameters(), b.module.parameters(), strict=True):
        torch.testing.assert_close(pa, pb)


@pytest.mark.parametrize("cls", NN_MODULE_ONLY)
def test_not_an_nn_module(cls, schema):
    assert nn.Module not in cls.__mro__
    learner = _make(cls, schema)
    assert not isinstance(learner, nn.Module)
    for name in ("eval", "cuda", "half", "forward", "apply", "parameters"):
        assert not hasattr(learner, name)
