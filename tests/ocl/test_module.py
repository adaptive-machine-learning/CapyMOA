"""Tests for the ``module`` interface of torch-backed OCL strategies."""

import numpy as np
import pytest
import torch
from torch import nn

from capymoa.core.torch.ann import Perceptron
from capymoa.ocl.datasets import TinySplitMNIST
from capymoa.ocl.strategy import (
    DER,
    DERPP,
    EWC,
    LWF,
    MAS,
    NCM,
    SCR,
    SI,
    SLDA,
    GDumb,
    RanPAC,
    RWalk,
)
from capymoa.ocl.strategy.l2p import L2P, L2PViT

pytestmark = pytest.mark.torch


class _ScrModel(nn.Module):
    def __init__(self, n_in: int):
        super().__init__()
        self.feature_extractor = nn.Sequential(
            nn.Flatten(), nn.Linear(n_in, 16), nn.ReLU()
        )
        self.projection = nn.Linear(16, 8)


class _DummyViT(L2PViT, nn.Module):
    def get_embedding_size(self) -> int:
        return 8

    def get_patch_embed(self, pixel_values):
        return torch.zeros((pixel_values.size(0), 4, 8), device=pixel_values.device)

    def forward_encoder(self, prompts, patch_embed):
        return torch.zeros((patch_embed.size(0), 4, 8), device=patch_embed.device)

    def forward_query(self, patch_embed):
        return torch.zeros((patch_embed.size(0), 8), device=patch_embed.device)


def _make(cls, schema, device="cpu"):
    if cls is NCM:
        return NCM(schema, device=device)
    if cls is SLDA:
        return SLDA(schema, device=device)
    if cls is RanPAC:
        return RanPAC(schema, projection_dim=16, device=device)
    if cls is GDumb:
        model = Perceptron(schema)
        return GDumb(schema, model, 1, 8, 32, device=device)
    if cls is L2P:
        mask = TinySplitMNIST().task_mask
        return L2P(schema, mask, vit=_DummyViT(), device=device)
    if cls is SCR:
        model = _ScrModel(schema.get_num_attributes())
        opt = torch.optim.SGD(model.parameters(), lr=0.1)
        return SCR(
            schema,
            model,
            opt,
            nn.Identity(),
            buffer_capacity=64,
            replay_batch_size=8,
            device=device,
        )
    model = Perceptron(schema)
    opt = torch.optim.SGD(model.parameters(), lr=0.1)
    if cls in (EWC, MAS, RWalk, SI):
        return cls(schema, model, opt, lambda_=1.0, device=device)
    return cls(schema, model, opt, device=device)


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


ALL = [EWC, LWF, MAS, RWalk, SI, DER, DERPP, SCR, NCM, SLDA, RanPAC, GDumb, L2P]


def _tensors(module: nn.Module):
    return [*module.parameters(), *module.buffers()]


@pytest.mark.parametrize("cls", ALL)
def test_meta_device_everywhere(cls, schema):
    learner = _make(cls, schema, device="meta")
    assert learner.device == torch.device("meta")
    tensors = _tensors(learner.module)
    assert tensors
    assert all(t.device.type == "meta" for t in tensors)


@pytest.mark.parametrize("cls", ALL)
def test_module_to_updates_device(cls, schema):
    learner = _make(cls, schema)
    assert learner.device == torch.device("cpu")
    learner.module.to("meta")
    assert learner.device == torch.device("meta")


# Strategies that need real values (``.item()``, indexing) cannot run on meta.
META_XFAIL = {
    DERPP: "finite loss check calls .item()",
    SCR: "loss logging calls .item()",
    NCM: "boolean mask indexing needs real values",
    SLDA: "boolean mask indexing needs real values",
    RanPAC: "boolean mask indexing needs real values",
    L2P: "loss logging calls .item()",
}


def _meta_params():
    for cls in ALL:
        marks = []
        if cls in META_XFAIL:
            marks.append(pytest.mark.xfail(strict=True, reason=META_XFAIL[cls]))
        yield pytest.param(cls, marks=marks, id=cls.__name__)


@pytest.mark.parametrize("cls", list(_meta_params()))
def test_meta_forward(cls, schema):
    learner = _make(cls, schema, device="meta")
    n = schema.get_num_attributes()
    x = torch.empty(4, n, device="meta", dtype=learner.x_dtype)
    y = torch.empty(4, device="meta", dtype=learner.y_dtype)
    learner.batch_predict_proba(x)
    learner.batch_train(x, y)


class _NoNumpyWithoutCpu(torch.Tensor):
    """Tensor whose ``numpy`` fails unless ``cpu`` ran first, like a GPU tensor."""

    _on_cpu = False

    def cpu(self, *args, **kwargs):
        out = super().cpu(*args, **kwargs).as_subclass(_NoNumpyWithoutCpu)
        out._on_cpu = True
        return out

    def numpy(self, *args, **kwargs):
        if not self._on_cpu:
            raise TypeError("can't convert device tensor to numpy, use .cpu() first")
        return super().numpy(*args, **kwargs)


def test_base_predict_proba_moves_to_cpu(schema):
    from capymoa.base import BatchClassifier
    from capymoa.core import Instance

    class Stub(BatchClassifier):
        def batch_train(self, x, y):
            pass

        def batch_predict_proba(self, x):
            out = torch.full((x.shape[0], 2), 0.5).as_subclass(_NoNumpyWithoutCpu)
            return out.requires_grad_()

    learner = Stub(schema)
    instance = Instance.from_array(schema, np.zeros(schema.get_num_attributes()))
    proba = learner.predict_proba(instance)
    assert proba.dtype == np.float64
    assert proba.tolist() == [0.5, 0.5]


def test_wrappers_follow_inner_learner(schema):
    from capymoa.classifier import Finetune
    from capymoa.ocl.strategy import RAR, ExperienceReplay

    model = Perceptron(schema).to(torch.float64)
    inner = Finetune(schema, model, device="meta")
    assert inner.x_dtype == torch.float64
    for wrapper in (
        ExperienceReplay(inner),
        RAR(inner, augment=nn.Identity()),
    ):
        assert wrapper.device == torch.device("meta")
        assert wrapper.x_dtype == torch.float64
        assert wrapper.y_dtype == inner.y_dtype
