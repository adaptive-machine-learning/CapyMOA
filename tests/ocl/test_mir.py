import pytest

pytest.importorskip("torch")

import torch
from torch import nn

from capymoa.ocl.strategy import MIR
from capymoa.stream import Schema

pytestmark = pytest.mark.torch


@pytest.mark.parametrize(
    "make",
    [
        lambda p: torch.optim.SGD(p, lr=0.1, momentum=0.9, weight_decay=0.1),
        lambda p: torch.optim.SGD(p, lr=0.1, momentum=0.9, nesterov=True),
        lambda p: torch.optim.Adam(p, lr=0.01),
        lambda p: torch.optim.AdamW(p, lr=0.01),
    ],
)
@pytest.mark.parametrize("history", [False, True])
def test_mir_scores_match_step_and_restore_state(make, history):
    """Scores equal the loss change of a real step, and nothing is left changed."""
    schema = Schema.from_custom(["x0", "x1", "class"], "class", {"class": ["a", "b"]})
    torch.manual_seed(12)
    model = nn.Sequential(nn.Linear(2, 4), nn.Tanh(), nn.Linear(4, 2))
    optimiser = make(model.parameters())
    learner = MIR(schema, model, optimiser)
    x, y = torch.randn(8, 2), torch.arange(8) % 2
    cx, cy = torch.randn(5, 2), torch.arange(5) % 2
    ce = nn.functional.cross_entropy
    if history:
        ce(model(x), y).backward()
        optimiser.step()
    for p in model.parameters():
        p.grad = torch.ones_like(p)
    weights = [p.clone() for p in model.parameters()]
    state = (
        {k: v.clone() for k, v in optimiser.state_dict()["state"][0].items()}
        if history
        else {}
    )

    scores = learner._interference_scores(x, y, cx, cy)

    for p, w in zip(model.parameters(), weights):
        torch.testing.assert_close(p, w, rtol=0, atol=0)
        torch.testing.assert_close(p.grad, torch.ones_like(p), rtol=0, atol=0)
    for k, v in state.items():
        torch.testing.assert_close(optimiser.state_dict()["state"][0][k], v)

    with torch.no_grad():
        before = ce(model(cx), cy, reduction="none")
    optimiser.zero_grad()
    ce(model(x), y).backward()
    optimiser.step()
    with torch.no_grad():
        after = ce(model(cx), cy, reduction="none")
    torch.testing.assert_close(scores, after - before)


def test_mir_restores_module_modes():
    schema = Schema.from_custom(["x0", "x1", "class"], "class", {"class": ["a", "b"]})
    model = nn.Sequential(nn.Linear(2, 2), nn.Dropout(0.5))
    learner = MIR(schema, model, torch.optim.SGD(model.parameters(), lr=0.1))
    model[1].eval()
    learner.batch_predict_proba(torch.ones(3, 2))
    assert model.training and not model[1].training
