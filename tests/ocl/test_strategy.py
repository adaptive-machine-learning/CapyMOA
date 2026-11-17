import inspect
from collections.abc import Callable
from dataclasses import asdict, dataclass
from functools import partial

import numpy as np
import pytest

pytestmark = pytest.markskip("torch")

import torch
from torch import nn

from capymoa.base import Classifier, TorchMixin
from capymoa.classifier import Finetune, HoeffdingTree
from capymoa.core.torch.ann import Perceptron
from capymoa.ocl import evaluate_ocl
from capymoa.ocl.datasets import TinySplitMNIST
from capymoa.ocl.strategy import (
    DER,
    DERPP,
    EWC,
    LWF,
    MAS,
    MIR,
    NCM,
    RAR,
    SCR,
    SI,
    SLDA,
    ExperienceReplay,
    GDumb,
    RanPAC,
    RWalk,
)
from capymoa.ocl.strategy.l2p import L2P, L2PViT
from capymoa.stream import Schema

# PyTorch is notorious for non-deterministic behavior between versions and platforms.
# Here we set a fixed absolute tolerance of +-1.5 for percentage-based metrics.
approx = partial(pytest.approx, abs=1.5)


SEED = 0
"""Seed for every learner that takes one, so results are exact."""


@dataclass(frozen=True)
class Result:
    accuracy_final: float
    anytime_accuracy_all_avg: float
    ttt_accuracy: float


@dataclass(frozen=True)
class Case:
    name: str
    constructor: Callable[..., Classifier]
    # `None` skips the accuracy check, the shared torch checks still run.
    expected: Result | None
    batch_size: int = 32
    epochs: int = 1
    continual_evaluations: int = 1
    task_mask: bool = False
    # Learner uses `TorchMixin`, so the shared module checks apply.
    torch: bool = True
    # Meta forward pass cannot run, with the reason (strict xfail).
    meta_xfail: str | None = None
    # State outside ``module`` changes predictions after a load (strict xfail).
    restore_xfail: str | None = None


def pre_processor() -> nn.Module:
    """Create a pre-processor for the schema."""
    torch.manual_seed(0)
    return nn.Sequential(
        nn.Linear(256, 512),
        nn.ReLU(),
    )


def new_constructor(
    learner: type[Classifier],
    lr: float,
    optimiser_type: type[torch.optim.Optimizer] = torch.optim.SGD,
    **kwargs,
) -> Callable[[Schema], Classifier]:
    """Create a new learner instance with the given hyperparameters."""

    def constructor(
        schema: Schema, random_seed: int | None = None, **extra_kwargs
    ) -> Classifier:
        if "random_seed" in inspect.signature(learner).parameters:
            extra_kwargs["random_seed"] = random_seed
        model = Perceptron(schema)
        optimiser = optimiser_type(model.parameters(), lr=lr)
        return learner(
            schema,
            model,
            optimiser=optimiser,  # type: ignore
            **kwargs,
            **extra_kwargs,
        )

    return constructor


def _new_rar(schema, random_seed=None, **kwargs):
    # RAR test case constructor
    return RAR(
        Finetune(schema, Perceptron(schema), random_seed=random_seed, **kwargs),
        augment=nn.Dropout(p=0.2),
        repeats=2,
    )


def _new_der(schema: Schema, **kwargs) -> DER:
    model = Perceptron(schema)
    return DER(schema, model, torch.optim.Adam(model.parameters(), lr=0.001), **kwargs)


def _new_scr(schema: Schema, **kwargs) -> SCR:
    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.feature_extractor = nn.Sequential(
                nn.Flatten(), nn.Linear(schema.get_num_attributes(), 64), nn.ReLU()
            )
            self.projection = nn.Sequential(
                nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, 32)
            )

    model = Model()
    return SCR(
        schema,
        model,
        torch.optim.SGD(model.parameters(), lr=0.1),
        augment=nn.Dropout(p=0.2),
        buffer_capacity=200,
        replay_batch_size=32,
        **kwargs,
    )


class _DummyViT(L2PViT, nn.Module):
    def get_embedding_size(self) -> int:
        return 8

    def get_patch_embed(self, pixel_values):
        return torch.zeros((pixel_values.size(0), 4, 8), device=pixel_values.device)

    def forward_encoder(self, prompts, patch_embed):
        return torch.zeros((patch_embed.size(0), 4, 8), device=patch_embed.device)

    def forward_query(self, patch_embed):
        return torch.zeros((patch_embed.size(0), 8), device=patch_embed.device)


def _new_l2p(schema: Schema, **kwargs) -> L2P:
    return L2P(schema, TinySplitMNIST().task_mask, vit=_DummyViT(), **kwargs)


"""
Add new test cases here.

Use the `partial` function to create a new function with hyperparameters already
set.
"""
TEST_CASES: list[Case] = [
    Case("MIR", new_constructor(MIR, lr=0.01), Result(13.5, 14.0, 7.1)),
    # These improved when `MOAClassifier.predict_proba` stopped discarding
    # predictions whose unnormalised vote total was below 1e-2. HoeffdingTree
    # uses Naive Bayes at its leaves, whose likelihood products fall below that
    # threshold, so most of its predictions were being thrown away and scored
    # as misses.
    Case(
        "HoeffdingTree",
        HoeffdingTree,
        Result(69.50, 45.50, 56.3),
        batch_size=1,
        torch=False,
    ),
    Case(
        "HoeffdingTree",
        HoeffdingTree,
        Result(69.50, 45.70, 51.0),
        batch_size=32,
        torch=False,
    ),
    Case(
        "RAR",
        _new_rar,
        Result(47.50, 29.20, 8.10),
        meta_xfail="inner Finetune NaN check needs a real value",
    ),
    Case(
        "Finetune",
        partial(Finetune, model=Perceptron),
        Result(27.00, 24.70, 8.3),
        epochs=2,
        continual_evaluations=2,
        meta_xfail="NaN check on the loss needs a real value",
    ),
    Case(
        "SLDA",
        SLDA,
        Result(80.5, 51.4, 63.0),
        meta_xfail="boolean mask indexing needs real values",
    ),
    Case(
        "SLDA_with_preprocessor",
        lambda s, **kw: SLDA(s, pre_processor(), 512, **kw),
        Result(87.5, 54.5, 67.7),
        meta_xfail="boolean mask indexing needs real values",
    ),
    Case(
        "RanPAC",
        lambda s, random_seed=None, **kw: RanPAC(
            s, projection_dim=512, random_seed=random_seed, **kw
        ),
        Result(78.0, 46.7, 60.8),
        meta_xfail="boolean mask indexing needs real values",
        restore_xfail="sample count is a plain int and weights are cached",
    ),
    Case(
        "NCM",
        NCM,
        Result(71.5, 46.2, 67.5),
        meta_xfail="boolean mask indexing needs real values",
    ),
    Case(
        "NCM_with_preprocessor",
        lambda s, **kw: NCM(s, pre_processor(), 512, **kw),
        Result(69.5, 45.0, 66.8),
        meta_xfail="boolean mask indexing needs real values",
    ),
    Case(
        "ExperienceReplay",
        lambda schema, random_seed=None, **kw: ExperienceReplay(
            Finetune(schema, Perceptron, random_seed=random_seed, **kw)
        ),
        Result(28.5, 20.1, 3.0),
        meta_xfail="inner Finetune NaN check needs a real value",
    ),
    Case(
        "DER",
        _new_der,
        Result(29.5, 20.5, 2.9),
    ),
    Case(
        "SCR",
        _new_scr,
        Result(65.0, 44.8, 63.6),
        meta_xfail="loss logging calls .item()",
        restore_xfail="replay buffer count is a plain int",
    ),
    Case(
        "DERPP",
        new_constructor(
            DERPP,
            lr=0.001,
            optimiser_type=torch.optim.Adam,
        ),
        Result(28.5, 20.0, 3.1),
        meta_xfail="finite loss check calls .item()",
    ),
    Case(
        "GDumb",
        lambda schema, **kw: GDumb(schema, Perceptron(schema), 2, 32, 200, **kw),
        # Changed from (40.5, 26.6, 0.0) when the offline fit's DataLoader
        # shuffle became seed-controlled instead of drawing from the global
        # torch RNG; the new values are fully deterministic.
        Result(37.5, 27.0, 0.0),
    ),
    Case(
        "L2P",
        _new_l2p,
        None,  # Accuracy needs the real ViT, see test_l2p.py.
        meta_xfail="loss logging calls .item()",
    ),
    Case(
        "EWC",
        new_constructor(EWC, lr=0.06, lambda_=313, gamma=0.91),
        Result(24.5, 18.8, 14.6),
    ),
    Case(
        "EWC(mask_test=True)",
        new_constructor(EWC, lr=0.06, lambda_=168.69, mask_test=True),
        Result(71.99, 47.20, 25.3),
        task_mask=True,
    ),
    Case(
        "SI",
        new_constructor(SI, lr=0.004, lambda_=96, optimiser_type=torch.optim.Adam),
        Result(38.5, 25.8, 16.1),
    ),
    Case(
        "SI(mask_train=True, mask_test=True)",
        new_constructor(
            SI,
            lr=0.006,
            lambda_=2.5,
            optimiser_type=torch.optim.Adam,
            mask_train=True,
            mask_test=True,
        ),
        Result(96.5, 80.8, 34.4),
        task_mask=True,
    ),
    Case(
        "LWF",
        new_constructor(LWF, lr=0.10, alpha=4.66, temperature=1.67),
        Result(36.49, 25.09, 17.59),
    ),
    Case(
        "MAS",
        new_constructor(
            MAS, optimiser_type=torch.optim.Adam, lr=0.004, lambda_=30, alpha=0.5
        ),
        Result(45.5, 35.8, 17.4),
    ),
    Case(
        "MAS(task_mask)",
        new_constructor(
            MAS,
            optimiser_type=torch.optim.Adam,
            lr=0.004,
            lambda_=30,
            alpha=0.5,
            mask_train=True,
            mask_test=True,
        ),
        Result(96.0, 81.2, 31.8),
        task_mask=True,
    ),
    Case(
        "RWalk",
        new_constructor(
            RWalk,
            optimiser_type=torch.optim.Adam,
            lr=0.0037,
            lambda_=53.4,
            alpha=0.84,
            delta_t=51,
        ),
        Result(43.0, 28.9, 10.8),
    ),
    Case(
        "RWalk(task_mask)",
        new_constructor(
            RWalk,
            optimiser_type=torch.optim.Adam,
            lr=0.0037,
            lambda_=53.4,
            alpha=0.84,
            delta_t=51,
            mask_train=True,
            mask_test=True,
        ),
        Result(87.5, 76.9, 31.1),
        task_mask=True,
    ),
]


def _params(cases: list[Case]):
    return [pytest.param(c, id=c.name) for c in cases]


ACCURACY_CASES = _params([c for c in TEST_CASES if c.expected is not None])
# Every case that uses `TorchMixin` gets the shared module checks below.
TORCH_CASES = _params([c for c in TEST_CASES if c.torch])
META_CASES = [
    pytest.param(
        c,
        id=c.name,
        marks=[pytest.mark.xfail(strict=True, reason=c.meta_xfail)]
        if c.meta_xfail
        else [],
    )
    for c in TEST_CASES
    if c.torch
]
RESTORE_CASES = [
    pytest.param(
        c,
        id=c.name,
        marks=[pytest.mark.xfail(strict=True, reason=c.restore_xfail)]
        if c.restore_xfail
        else [],
    )
    for c in TEST_CASES
    if c.torch
]


@pytest.mark.parametrize("case", ACCURACY_CASES)
def test_ocl_classifier(case: Case):
    assert case.expected is not None
    scenario = TinySplitMNIST()

    # Set random seeds for reproducibility
    torch.manual_seed(0)
    np.random.seed(0)
    torch.use_deterministic_algorithms(True)

    kwargs = {}
    if "random_seed" in inspect.signature(case.constructor).parameters:
        kwargs["random_seed"] = SEED
    if case.task_mask:
        kwargs["task_mask"] = scenario.task_mask
    learner = case.constructor(scenario.schema, **kwargs)  # type: ignore
    r = evaluate_ocl(
        learner,
        scenario.train_loaders(case.batch_size),
        scenario.test_loaders(case.batch_size),
        epochs=case.epochs,
    )
    actual = Result(
        r["accuracy_final"] * 100,
        r["anytime_accuracy_all_avg"] * 100,
        r["ttt"]["cumulative"]["accuracy"],
    )
    assert asdict(actual) == approx(asdict(case.expected)), f"Case {case.name} failed."

    def assert_ndarray(value: np.ndarray, shape: tuple, dtype: type = np.float32):
        assert isinstance(value, np.ndarray), (
            f"Expected np.ndarray but got {type(value)}"
        )
        assert value.shape == shape, f"Expected shape {shape} but got {value.shape}"

    def assert_float(value: float):
        assert isinstance(value, float), f"Expected float but got {type(value)}"

    n_tasks = r["n_tasks"]
    total_eval = r["n_continual_evaluations"] * n_tasks
    assert_float(r["accuracy_all_avg"])
    assert_float(r["accuracy_final"])
    assert_float(r["accuracy_seen_avg"])
    assert_float(r["anytime_accuracy_all_avg"])
    assert_float(r["anytime_accuracy_seen_avg"])
    assert_float(r["backward_transfer"])
    assert_float(r["forward_transfer"])
    assert_ndarray(r["accuracy_matrix"], (n_tasks, n_tasks))
    assert_ndarray(r["anytime_accuracy_matrix"], (total_eval, n_tasks))
    assert_ndarray(r["boundaries"], (n_tasks + 1,), dtype=np.integer)
    assert_ndarray(
        r["class_cm"], (n_tasks, r["n_classes"], r["n_classes"]), dtype=np.integer
    )
    assert list(r["per_task"]) == ["task", "accuracy_all", "accuracy_seen"]
    assert len(r["per_task"]["task"]) == n_tasks
    assert list(r["anytime"]) == [
        "task",
        "step",
        "accuracy_all",
        "accuracy_seen",
    ]
    assert len(r["anytime"]["task"]) == total_eval
    assert "task" in r["ttt"]["windowed"]


def _build(case: Case, **kwargs) -> Classifier:
    scenario = TinySplitMNIST()
    if case.task_mask:
        kwargs["task_mask"] = scenario.task_mask
    torch.manual_seed(0)
    return case.constructor(scenario.schema, **kwargs)


def _tensors(module: nn.Module) -> list[torch.Tensor]:
    return [*module.parameters(), *module.buffers()]


def _batch(learner, n: int = 16):
    schema = learner.schema
    x = torch.rand(n, schema.get_num_attributes(), dtype=learner.x_dtype)
    y = torch.randint(0, schema.get_num_classes(), (n,), dtype=learner.y_dtype)
    return x, y


@pytest.mark.parametrize("case", TORCH_CASES)
def test_module_access(case: Case):
    learner = _build(case)
    assert isinstance(learner, TorchMixin)
    assert isinstance(learner.module, nn.Module)
    assert learner.module is learner._module
    with pytest.raises(AttributeError):
        learner.module = nn.Linear(1, 1)  # type: ignore
    if hasattr(learner.module, "model"):
        assert isinstance(learner.module.model, nn.Module)


@pytest.mark.parametrize("case", TORCH_CASES)
def test_no_nn_module_surface(case: Case):
    learner = _build(case)
    assert nn.Module not in type(learner).__mro__
    assert not isinstance(learner, nn.Module)
    names = ("eval", "cuda", "half", "forward", "apply", "parameters")
    names += ("state_dict", "load_state_dict", "model", "optimizer", "criterion")
    for name in names:
        assert not hasattr(learner, name), name


@pytest.mark.parametrize("case", RESTORE_CASES)
def test_state_dict_round_trip(case: Case):
    """A checkpoint of ``module`` holds all state needed to predict."""
    trained, other = _build(case), _build(case)
    x, y = _batch(trained)
    for _ in range(3):
        trained.batch_train(x, y)
    # Train ``other`` on different data, so a load that misses state shows up.
    other.batch_train(*_batch(other, n=8))
    other.module.load_state_dict(trained.module.state_dict())
    with torch.no_grad():
        torch.testing.assert_close(
            other.batch_predict_proba(x), trained.batch_predict_proba(x)
        )


@pytest.mark.parametrize("case", TORCH_CASES)
def test_meta_device_construction(case: Case):
    learner = _build(case, device="meta")
    assert learner.device == torch.device("meta")
    tensors = _tensors(learner.module)
    assert tensors
    assert all(t.device.type == "meta" for t in tensors)


@pytest.mark.parametrize("case", TORCH_CASES)
def test_module_to_updates_device(case: Case):
    learner = _build(case)
    assert learner.device == torch.device("cpu")
    learner.module.to("meta")
    assert learner.device == torch.device("meta")
    assert all(t.device.type == "meta" for t in _tensors(learner.module))


@pytest.mark.parametrize("case", META_CASES)
def test_meta_forward(case: Case):
    """Best effort: strategies that need real values are strict xfail."""
    learner = _build(case, device="meta")
    n = learner.schema.get_num_attributes()
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


def test_base_predict_proba_moves_to_cpu():
    from capymoa.base import BatchClassifier
    from capymoa.core import Instance

    class Stub(BatchClassifier):
        def batch_train(self, x, y):
            pass

        def batch_predict_proba(self, x):
            out = torch.full((x.shape[0], 2), 0.5).as_subclass(_NoNumpyWithoutCpu)
            return out.requires_grad_()

    schema = TinySplitMNIST().schema
    learner = Stub(schema)
    instance = Instance.from_array(schema, np.zeros(schema.get_num_attributes()))
    proba = learner.predict_proba(instance)
    assert proba.dtype == np.float64
    assert proba.tolist() == [0.5, 0.5]


def _wrapper(cls, inner):
    return cls(inner, augment=nn.Identity()) if cls is RAR else cls(inner)


@pytest.mark.parametrize("cls", [ExperienceReplay, RAR])
def test_wrapper_follows_inner_learner(cls):
    schema = TinySplitMNIST().schema
    model = Perceptron(schema).to(torch.float64)
    inner = Finetune(schema, model, device="meta")
    wrapper = _wrapper(cls, inner)
    assert wrapper.device == torch.device("meta")
    assert wrapper.x_dtype == torch.float64
    assert wrapper.y_dtype == inner.y_dtype


@pytest.mark.parametrize("cls", [ExperienceReplay, RAR])
def test_wrapper_module_moves_inner_learner_and_buffer(cls):
    schema = TinySplitMNIST().schema
    wrapper = _wrapper(cls, Finetune(schema, Perceptron(schema)))
    wrapper.module.to("meta")
    assert wrapper.device == torch.device("meta")
    inner = wrapper.module.learner
    assert all(p.device.type == "meta" for p in inner.parameters())
    buf = wrapper.module.buffer if cls is ExperienceReplay else wrapper.module.coreset
    assert all(b.device.type == "meta" for b in buf.buffers())
    assert wrapper.learner.device == torch.device("meta")


@pytest.mark.parametrize("cls", [ExperienceReplay, RAR])
def test_wrapper_needs_torch_mixin(cls):
    with pytest.raises(TypeError):
        _wrapper(cls, HoeffdingTree(TinySplitMNIST().schema))
