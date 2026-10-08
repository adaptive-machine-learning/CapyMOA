import pytest


def _torch_api():
    torch = pytest.importorskip("torch")
    from capymoa.core.torch import resolve_model, resolve_optimizer
    from capymoa.core.torch.ann import Perceptron

    return torch, resolve_model, resolve_optimizer, Perceptron


def test_resolve_model_name():
    _, resolve_model, _, Perceptron = _torch_api()
    assert resolve_model("Perceptron") is Perceptron


def test_resolve_model_passthrough():
    torch, resolve_model, _, _ = _torch_api()
    model = torch.nn.Identity()
    assert resolve_model(model) is model
    assert resolve_model(torch.nn.Identity) is torch.nn.Identity


def test_resolve_model_unknown_name():
    _, resolve_model, _, _ = _torch_api()
    with pytest.raises(ValueError, match="Unknown model 'MissingModel'"):
        resolve_model("MissingModel")


def test_resolve_optimizer_name():
    torch, _, resolve_optimizer, _ = _torch_api()
    assert resolve_optimizer("Adam") is torch.optim.Adam


def test_resolve_optimizer_passthrough():
    torch, _, resolve_optimizer, _ = _torch_api()
    optimizer = torch.optim.SGD([torch.nn.Parameter(torch.zeros(1))], lr=0.1)
    assert resolve_optimizer(optimizer) is optimizer
    assert resolve_optimizer(torch.optim.SGD) is torch.optim.SGD


def test_resolve_optimizer_unknown_name():
    _, _, resolve_optimizer, _ = _torch_api()
    with pytest.raises(ValueError, match="Unknown optimizer 'MissingOptimizer'"):
        resolve_optimizer("MissingOptimizer")


def test_nested_lazy_classifier_from_params(monkeypatch):
    pytest.importorskip("torch")
    import sys

    import capymoa.classifier as classifier_module
    from capymoa.base import LearnerParamsMixin
    from capymoa.base._learner_params import _LEARNER_REGISTRY
    from capymoa.datasets import ElectricityTiny

    class _NestedLearnerHolder(LearnerParamsMixin):
        def __init__(self, child=None, schema=None, random_seed=1):
            self.child = child

    # Simulate first access to the lazy Finetune export, including its registry.
    monkeypatch.delattr(classifier_module, "Finetune", raising=False)
    monkeypatch.delitem(sys.modules, "capymoa.classifier._finetune", raising=False)
    monkeypatch.delitem(
        _LEARNER_REGISTRY._registered,
        "capymoa.classifier.Finetune",
        raising=False,
    )

    schema = ElectricityTiny().get_schema()
    learner = _NestedLearnerHolder.from_params(
        schema,
        {
            "child": {
                "learner": "capymoa.classifier.Finetune",
                "params": {
                    "model": "Perceptron",
                    "optimizer": "Adam",
                    "optimizer_params": {"lr": 0.01},
                },
            }
        },
    )

    assert type(learner.child).__name__ == "Finetune"
