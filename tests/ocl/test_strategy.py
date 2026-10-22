from collections.abc import Callable
from dataclasses import asdict, dataclass
from functools import partial

import numpy as np
import pytest

pytestmark = pytest.markskip("torch")

import torch
from torch import nn

from capymoa.base import Classifier
from capymoa.classifier import Finetune, HoeffdingTree
from capymoa.core import Instance, LabeledInstance
from capymoa.core.torch.ann import Perceptron
from capymoa.ocl import evaluate_ocl
from capymoa.ocl.datasets import TinySplitMNIST
from capymoa.ocl.strategy import (
    DER,
    DERPP,
    EWC,
    LWF,
    MAS,
    NCM,
    PCR,
    RAR,
    SI,
    SLDA,
    ExperienceReplay,
    GDumb,
    RanPAC,
    RWalk,
)
from capymoa.ocl.strategy._pcr import _ProxyContrastiveLoss
from capymoa.stream import Schema

# PyTorch is notorious for non-deterministic behavior between versions and platforms.
# Here we set a fixed absolute tolerance of +-1.5 for percentage-based metrics.
approx = partial(pytest.approx, abs=1.5)


@dataclass(frozen=True)
class Result:
    accuracy_final: float
    anytime_accuracy_all_avg: float
    ttt_accuracy: float


@dataclass(frozen=True)
class Case:
    name: str
    constructor: Callable[[Schema], Classifier]
    expected: Result
    batch_size: int = 32
    epochs: int = 1
    continual_evaluations: int = 1
    task_mask: bool = False


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

    def constructor(schema: Schema, **extra_kwargs) -> Classifier:
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


def _new_rar(schema):
    # RAR test case constructor
    return RAR(
        Finetune(schema, Perceptron(schema)),
        augment=nn.Dropout(p=0.2),
        repeats=2,
    )


def _new_der(schema: Schema) -> DER:
    model = Perceptron(schema)
    return DER(schema, model, torch.optim.Adam(model.parameters(), lr=0.001))


def _new_pcr(schema: Schema) -> PCR:
    encoder = nn.Sequential(
        nn.Flatten(), nn.Linear(schema.get_num_attributes(), 64), nn.ReLU()
    )
    return PCR(
        schema,
        encoder,
        feature_dim=64,
        augment=nn.Dropout(p=0.2),
        replay_batch_size=32,
    )


"""
Add new test cases here.

Use the `partial` function to create a new function with hyperparameters already
set.
"""
TEST_CASES: list[Case] = [
    Case("PCR", _new_pcr, Result(69.5, 46.5, 46.6)),
    # These improved when `MOAClassifier.predict_proba` stopped discarding
    # predictions whose unnormalised vote total was below 1e-2. HoeffdingTree
    # uses Naive Bayes at its leaves, whose likelihood products fall below that
    # threshold, so most of its predictions were being thrown away and scored
    # as misses.
    Case("HoeffdingTree", HoeffdingTree, Result(69.50, 45.50, 56.3), batch_size=1),
    Case("HoeffdingTree", HoeffdingTree, Result(69.50, 45.70, 51.0), batch_size=32),
    Case("RAR", _new_rar, Result(47.50, 29.20, 8.10)),
    Case(
        "Finetune",
        partial(Finetune, model=Perceptron),
        Result(27.00, 24.70, 8.3),
        epochs=2,
        continual_evaluations=2,
    ),
    Case("SLDA", SLDA, Result(80.5, 51.4, 63.0)),
    Case(
        "SLDA_with_preprocessor",
        lambda s: SLDA(s, pre_processor(), 512),
        Result(87.5, 54.5, 67.7),
    ),
    Case("RanPAC", lambda s: RanPAC(s, projection_dim=512), Result(80.0, 48.0, 62.4)),
    Case("NCM", NCM, Result(71.5, 46.2, 67.5)),
    Case(
        "NCM_with_preprocessor",
        lambda s: NCM(s, pre_processor(), 512),
        Result(69.5, 45.0, 66.8),
    ),
    Case(
        "ExperienceReplay",
        lambda schema: ExperienceReplay(Finetune(schema, Perceptron)),
        Result(28.5, 20.1, 3.0),
    ),
    Case(
        "DER",
        _new_der,
        Result(29.5, 20.5, 2.9),
    ),
    Case(
        "DERPP",
        new_constructor(
            DERPP,
            lr=0.001,
            optimiser_type=torch.optim.Adam,
        ),
        Result(28.5, 20.0, 3.1),
    ),
    Case(
        "GDumb",
        lambda schema: GDumb(schema, Perceptron(schema), 2, 32, 200),
        # Changed from (40.5, 26.6, 0.0) when the offline fit's DataLoader
        # shuffle became seed-controlled instead of drawing from the global
        # torch RNG; the new values are fully deterministic.
        Result(37.5, 27.0, 0.0),
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


@pytest.mark.parametrize("case", TEST_CASES, ids=[test.name for test in TEST_CASES])
def test_ocl_classifier(case: Case):
    scenario = TinySplitMNIST()

    # Set random seeds for reproducibility
    torch.manual_seed(0)
    np.random.seed(0)
    torch.use_deterministic_algorithms(True)

    kwargs = {}
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


def _tiny_pcr(**kwargs) -> PCR:
    schema = Schema.from_custom(
        ["x0", "x1", "class"], "class", {"class": ["a", "b", "c"]}
    )
    model = kwargs.pop("model", None)
    if model is None:
        model = nn.Linear(2, 2, bias=False)
    defaults = {
        "schema": schema,
        "model": model,
        "feature_dim": 2,
        "augment": lambda x: x.flip(-1),
        "buffer_capacity": 8,
        "replay_batch_size": 2,
        "seed": 4,
    }
    defaults.update(kwargs)
    return PCR(**defaults)


def test_pcr_proxy_loss():
    """Repeated labels weight the denominator; absent proxies get no gradient."""
    features = torch.tensor(
        [[1.0, 0.2], [0.3, 1.0], [-1.0, 0.4]], dtype=torch.float64, requires_grad=True
    )
    proxies = torch.tensor(
        [[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]], dtype=torch.float64, requires_grad=True
    )
    labels = torch.tensor([0, 0, 1])
    temperature = 0.3
    # Independent scalar calculation, including one proxy entry per example.
    expected = []
    for feature, label in zip(features, labels):
        similarities = [
            (feature @ proxy / (feature.norm() * proxy.norm())) / temperature
            for proxy in proxies
        ]
        denominator = sum(similarities[int(y)].exp() for y in labels)
        expected.append(denominator.log() - similarities[int(label)])
    loss = _ProxyContrastiveLoss(temperature)(features, labels, proxies)
    torch.testing.assert_close(loss, torch.stack(expected).mean())
    loss.backward()
    assert proxies.grad[:2].abs().sum() > 0
    torch.testing.assert_close(proxies.grad[2], torch.zeros(2, dtype=torch.float64))
    assert features.grad.abs().sum() > 0
    # Keeping the own-label proxy makes a one-class batch finite and constant.
    one_class = _ProxyContrastiveLoss(temperature)(
        features.detach(), torch.zeros(3, dtype=torch.long), proxies.detach()
    )
    torch.testing.assert_close(one_class, torch.tensor(3.0, dtype=torch.float64).log())


def test_pcr_replay_order_and_updates(monkeypatch):
    """First fill memory, then train both encoder and proxies on history only."""
    learner = _tiny_pcr(substeps=2, augment=lambda x: x.add_(0.25))
    x, y = torch.eye(2), torch.tensor([0, 1])
    before_encoder = learner._model.weight.detach().clone()
    before_proxies = learner._proxies.weight.detach().clone()
    learner.batch_train(x, y)
    assert learner.last_loss == 0.0
    torch.testing.assert_close(learner._model.weight, before_encoder)
    torch.testing.assert_close(learner._proxies.weight, before_proxies)
    counts = []
    original_sample = learner._buffer.sample

    def sample(n):
        counts.append(learner._buffer.count)
        return original_sample(n)

    monkeypatch.setattr(learner._buffer, "sample", sample)
    learner.batch_train(x + 2, torch.tensor([1, 2]))
    assert counts == [2, 2]
    assert learner._buffer.count == 4
    assert learner.last_loss > 0
    assert not torch.allclose(learner._model.weight, before_encoder)
    assert not torch.allclose(learner._proxies.weight, before_proxies)
    # In-place augmentation must not overwrite raw input views or stored data.
    torch.testing.assert_close(x, torch.eye(2))
    torch.testing.assert_close(learner._buffer.array()["x"], torch.cat((x, x + 2)))


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_pcr_inference_and_instance_api(dtype):
    learner = _tiny_pcr(model=nn.Linear(2, 2, bias=False).to(dtype))
    with torch.no_grad():
        learner._model.weight.copy_(torch.eye(2, dtype=dtype))
        learner._proxies.weight.copy_(
            torch.tensor([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]], dtype=dtype)
        )
    x = torch.tensor([[3.0, 0.1], [-3.0, 0.1]], dtype=dtype)
    learner._model.train()
    probabilities = learner.batch_predict_proba(x)
    assert learner._model.training
    expected = (
        nn.functional.normalize(x, dim=1) @ learner._proxies.weight.T / 0.09
    ).softmax(1)
    torch.testing.assert_close(probabilities, expected)
    assert probabilities.argmax(1).tolist() == [0, 2]
    instance = Instance.from_array(learner.schema, x[0].numpy())
    np.testing.assert_allclose(
        learner.predict_proba(instance), probabilities[0].numpy(), rtol=1e-5
    )
    learner.train(LabeledInstance.from_array(learner.schema, x[0].numpy(), 0))
    assert learner._buffer.count == 1


def test_pcr_optimizer_and_seed():
    model = nn.Linear(2, 2)
    optimiser = torch.optim.Adam(model.parameters(), lr=0.003)
    state = torch.random.get_rng_state().clone()
    learner = _tiny_pcr(model=model, optimiser=optimiser)
    torch.testing.assert_close(torch.random.get_rng_state(), state)
    owned = {id(p) for g in optimiser.param_groups for p in g["params"]}
    assert all(id(p) in owned for p in learner._proxies.parameters())
    factory_learner = _tiny_pcr(
        optimiser=lambda params: torch.optim.Adam(params, lr=0.003)
    )
    assert isinstance(factory_learner._optimiser, torch.optim.Adam)
    torch.testing.assert_close(learner._proxies.weight, factory_learner._proxies.weight)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"feature_dim": 0},
        {"buffer_capacity": 0},
        {"substeps": 0},
        {"replay_batch_size": 0},
        {"temperature": 0},
        {"temperature": float("nan")},
    ],
)
def test_pcr_invalid_configuration(kwargs):
    with pytest.raises(ValueError):
        _tiny_pcr(**kwargs)


@pytest.mark.parametrize(
    "interface", ["forward", "forward_features", "features", "custom"]
)
def test_pcr_shaped_encoder_interfaces(interface):
    class Encoder(nn.Module):
        def __init__(self):
            super().__init__()
            self.embedding = nn.Sequential(nn.Flatten(), nn.Linear(2, 4))

        def forward(self, x):
            assert x.shape[1:] == (1, 1, 2)
            return self.embedding(x)

    model = Encoder()
    kwargs = {}
    if interface in ("forward_features", "features"):
        setattr(model, interface, model.forward)
    elif interface == "custom":
        kwargs["feature_extractor"] = model.forward
    learner = _tiny_pcr(model=model, feature_dim=4, replay_batch_size=None, **kwargs)
    learner.schema.shape = (1, 1, 2)
    # Shape is fixed when a learner is constructed.
    learner = PCR(
        learner.schema, model, 4, lambda x: x.flip(-1), replay_batch_size=None, **kwargs
    )
    x, y = torch.eye(2), torch.tensor([0, 1])
    learner.batch_train(x, y)
    learner.batch_train(x.reshape(2, 1, 1, 2), y)
    assert learner.last_loss > 0
    probabilities = learner.batch_predict_proba(x)
    assert probabilities.shape == (2, 3)
    torch.testing.assert_close(probabilities.sum(1), torch.ones(2))
