from collections.abc import Callable
from dataclasses import asdict, dataclass
from functools import partial
from itertools import permutations
from math import factorial

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
    ASER,
    DER,
    DERPP,
    EWC,
    LWF,
    MAS,
    NCM,
    RAR,
    SI,
    SLDA,
    ExperienceReplay,
    GDumb,
    RanPAC,
    RWalk,
)
from capymoa.ocl.strategy._aser import _knn_shapley
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


def _new_aser(schema: Schema) -> ASER:
    model = nn.Sequential(
        nn.Flatten(),
        nn.Linear(schema.get_num_attributes(), 50),
        nn.ReLU(),
        nn.Linear(50, schema.get_num_classes()),
    )
    return ASER(schema, model, torch.optim.SGD(model.parameters(), lr=0.1))


"""
Add new test cases here.

Use the `partial` function to create a new function with hyperparameters already
set.
"""
TEST_CASES: list[Case] = [
    Case("ASER", _new_aser, Result(21.0, 18.8, 23.9)),
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
def test_aser_reference_passes_and_prediction(dtype):
    model = nn.Sequential(
        nn.Linear(2, 4), nn.BatchNorm1d(4), nn.ReLU(), nn.Linear(4, 3)
    ).to(dtype)
    learner = _tiny_aser(model=model)
    x, y = torch.eye(2).repeat(2, 1), torch.tensor([0, 1, 0, 1])
    learner.batch_train(x, y)
    assert int(model[1].num_batches_tracked) == 2
    learner.batch_train(x, y)
    assert int(model[1].num_batches_tracked) == 5
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
    # A batch-normalized encoder cannot train on a singleton, as in the reference.
    single = _tiny_aser(model=nn.Linear(2, 3).to(dtype))
    single.train(LabeledInstance.from_array(single.schema, x[0].numpy(), 0))
    assert single._seen == 1


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
