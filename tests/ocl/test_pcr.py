import numpy as np
import pytest

pytestmark = pytest.markskip("torch")  # type: ignore[attr-defined]

import torch
from torch import nn
from torch.utils.data import TensorDataset

from capymoa.core import Instance, LabeledInstance
from capymoa.core.torch import FeatureModel
from capymoa.ocl.strategy import PCR
from capymoa.ocl.strategy._pcr import _ProxyContrastiveLoss
from capymoa.stream import Schema, TorchStream


class LinearFeatures(nn.Linear, FeatureModel):
    def features(self, x):
        return super().forward(x)


def make_pcr(**kwargs):
    schema = Schema.from_custom(
        ["x0", "x1", "class"], "class", {"class": ["a", "b", "c"]}
    )
    model = kwargs.pop("model", None)
    if model is None:
        model = LinearFeatures(2, 2, bias=False)
    return PCR(schema, model, 2, kwargs.pop("augment", lambda x: x), **kwargs)


def test_requires_feature_model():
    """Reject models that do not meet PCR's FeatureModel contract."""
    with pytest.raises(TypeError, match="model must be a FeatureModel"):
        make_pcr(model=nn.Linear(2, 2))


def test_feature_width_must_match_proxy_head():
    """Catch feature/proxy width mismatches before contrastive loss computation."""
    learner = make_pcr(model=LinearFeatures(2, 3))
    x, y = torch.eye(2), torch.tensor([0, 1])
    learner.batch_train(x, y)
    with pytest.raises(ValueError, match="encoder output width does not match"):
        learner.batch_train(x, y)


def test_seen_classes_and_state_dict():
    """Mask unseen classes and preserve that mask when module state is restored."""
    learner = make_pcr(buffer_capacity=1)
    x = torch.tensor([[1.0, 0.0]])
    torch.testing.assert_close(
        learner.batch_predict_proba(x), torch.full((1, 3), 1 / 3)
    )
    with torch.no_grad():
        learner.module.model.weight.copy_(torch.eye(2))
        learner.module.proxies.weight.copy_(
            torch.tensor([[-1.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
        )
    learner.batch_train(x, torch.tensor([0]))
    torch.testing.assert_close(
        learner.batch_predict_proba(x), torch.tensor([[1.0, 0.0, 0.0]])
    )
    learner.batch_train(x, torch.tensor([2]))
    scores = learner.batch_predict_proba(x)
    assert scores[0, 1] == 0
    assert scores[0, 0] > 0 and scores[0, 2] > 0
    restored = make_pcr(buffer_capacity=1)
    restored.module.load_state_dict(learner.module.state_dict())
    torch.testing.assert_close(restored.batch_predict_proba(x), scores)


def test_stream_augmentation_reused_across_substeps():
    """Reuse the stream view across steps and augment replay anew each step."""
    calls = []

    def augment(x):
        calls.append(x.clone())
        return x + len(calls)

    class Encoder(FeatureModel):
        def __init__(self):
            super().__init__()
            self.layer = nn.Linear(2, 2)
            self.inputs = []

        def features(self, x):
            self.inputs.append(x.detach().clone())
            return self.layer(x)

    model = Encoder()
    learner = make_pcr(model=model, augment=augment, substeps=3, replay_batch_size=1)
    learner.batch_train(torch.tensor([[1.0, 0.0]]), torch.tensor([0]))
    current = torch.tensor([[0.0, 1.0]])
    learner.batch_train(current, torch.tensor([1]))
    assert len(calls) == 4
    torch.testing.assert_close(calls[0], current)
    for batch in model.inputs:
        torch.testing.assert_close(batch[-1:], current + 1)


def _tiny_pcr(**kwargs) -> PCR:
    schema = Schema.from_custom(
        ["x0", "x1", "class"], "class", {"class": ["a", "b", "c"]}
    )
    model = kwargs.pop("model", None)
    if model is None:
        model = LinearFeatures(2, 2, bias=False)
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
    """Check duplicate-positive weighting and gradients against the loss formula."""
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
        denominator = torch.stack([similarities[int(y)].exp() for y in labels]).sum()
        expected.append(denominator.log() - similarities[int(label)])
    loss = _ProxyContrastiveLoss(temperature)(features, labels, proxies)
    torch.testing.assert_close(loss, torch.stack(expected).mean())
    loss.backward()
    assert features.grad is not None and proxies.grad is not None
    assert proxies.grad[:2].abs().sum() > 0
    torch.testing.assert_close(proxies.grad[2], torch.zeros(2, dtype=torch.float64))
    assert features.grad.abs().sum() > 0
    # Keeping the own-label proxy makes a one-class batch finite and constant.
    one_class = _ProxyContrastiveLoss(temperature)(
        features.detach(), torch.zeros(3, dtype=torch.long), proxies.detach()
    )
    torch.testing.assert_close(one_class, torch.tensor(3.0, dtype=torch.float64).log())


# Replay only history, update both parameter sets, then store the current batch.
def test_pcr_replay_order_and_updates(monkeypatch):
    """Replay history only, update encoder and proxies, then store the new batch."""
    learner = _tiny_pcr(substeps=2, augment=lambda x: x.add_(0.25))
    x, y = torch.eye(2), torch.tensor([0, 1])
    before_encoder = learner.module.model.weight.detach().clone()
    before_proxies = learner.module.proxies.weight.detach().clone()
    learner.batch_train(x, y)
    torch.testing.assert_close(learner.module.model.weight, before_encoder)
    torch.testing.assert_close(learner.module.proxies.weight, before_proxies)
    counts = []
    original_sample = learner.module.buffer.sample

    def sample(n):
        counts.append(learner.module.buffer.count)
        return original_sample(n)

    monkeypatch.setattr(learner.module.buffer, "sample", sample)
    learner.batch_train(x + 2, torch.tensor([1, 2]))
    assert counts == [2, 2]
    assert learner.module.buffer.count == 4
    assert not torch.allclose(learner.module.model.weight, before_encoder)
    assert not torch.allclose(learner.module.proxies.weight, before_proxies)
    # In-place augmentation must not overwrite raw input views or stored data.
    torch.testing.assert_close(x, torch.eye(2))
    torch.testing.assert_close(
        learner.module.buffer.array()["x"], torch.cat((x, x + 2))
    )


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_pcr_inference_and_instance_api(dtype):
    """Preserve cosine predictions and instance API behavior across dtypes."""
    learner = _tiny_pcr(model=LinearFeatures(2, 2, bias=False).to(dtype))
    learner.batch_train(
        torch.tensor([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]], dtype=dtype),
        torch.tensor([0, 1, 2]),
    )
    with torch.no_grad():
        learner.module.model.weight.copy_(torch.eye(2, dtype=dtype))
        learner.module.proxies.weight.copy_(
            torch.tensor([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]], dtype=dtype)
        )
    x = torch.tensor([[3.0, 0.1], [-3.0, 0.1]], dtype=dtype)
    learner.module.model.train()
    probabilities = learner.batch_predict_proba(x)
    assert not learner.module.model.training
    expected = (
        nn.functional.normalize(x, dim=1) @ learner.module.proxies.weight.T / 0.09
    ).softmax(1)
    torch.testing.assert_close(probabilities, expected)
    assert probabilities.argmax(1).tolist() == [0, 2]
    instance = Instance.from_array(learner.schema, x[0].numpy())
    np.testing.assert_allclose(
        learner.predict_proba(instance), probabilities[0].numpy(), rtol=1e-5
    )
    learner.train(LabeledInstance.from_array(learner.schema, x[0].numpy(), 0))
    assert learner.module.buffer.count == 4


def test_pcr_optimizer_class_and_kwargs():
    """Honor the selected optimizer class, keyword options, and defaults."""
    learner = _tiny_pcr(
        optimiser_class=torch.optim.Adam, optimiser_kwargs={"lr": 0.003}
    )
    assert isinstance(learner._optimiser, torch.optim.Adam)
    assert learner._optimiser.defaults["lr"] == 0.003
    default_learner = _tiny_pcr()
    assert isinstance(default_learner._optimiser, torch.optim.SGD)
    assert default_learner._optimiser.defaults["lr"] == 0.001


def test_pcr_resolves_seed_for_proxy_and_replay(monkeypatch):
    """Use the resolved seed consistently for proxies and reservoir sampling."""
    monkeypatch.setattr(np.random, "randint", lambda *args: 123)
    model = LinearFeatures(2, 2, bias=False)
    learner = make_pcr(model=model, buffer_capacity=3)
    reference = make_pcr(
        model=LinearFeatures(2, 2, bias=False), buffer_capacity=3, seed=123
    )
    assert learner.random_seed == reference.random_seed == 123
    torch.testing.assert_close(
        learner.module.proxies.weight, reference.module.proxies.weight
    )

    x = torch.arange(20, dtype=torch.float32).reshape(10, 2)
    y = torch.arange(10) % 3
    learner.batch_train(x, y)
    reference.batch_train(x, y)
    samples = learner.module.buffer.sample(8)
    reference_samples = reference.module.buffer.sample(8)
    for key in ("x", "y"):
        torch.testing.assert_close(
            learner.module.buffer.array()[key], reference.module.buffer.array()[key]
        )
        torch.testing.assert_close(samples[key], reference_samples[key])


def test_pcr_cnn_encoder_with_flat_and_image_inputs():
    """Reshape flat rows so CNN features receive image batches."""

    class CNN(FeatureModel):
        def __init__(self):
            super().__init__()
            self.conv = nn.Conv2d(1, 4, kernel_size=3, padding=1)
            self.pool = nn.AdaptiveAvgPool2d(1)
            self.classifier = nn.Linear(4, 2)

        def features(self, x):
            assert x.ndim == 4 and x.shape[1:] == (1, 8, 8)
            return self.pool(self.conv(x)).flatten(1)

        def forward(self, x):
            return self.classifier(self.features(x))

    schema = TorchStream.from_classification(
        TensorDataset(torch.zeros(3, 1, 8, 8), torch.tensor([0, 1, 0])),
        num_classes=2,
        shape=(1, 8, 8),
    ).get_schema()
    learner = PCR(schema, CNN(), feature_dim=4, augment=lambda x: x.flip(-1))
    x = torch.rand(3, 64)
    y = torch.tensor([0, 1, 0])

    # Batch and row APIs provide flattened vectors; the CNN sees NCHW tensors.
    learner.batch_train(x, y)
    learner.batch_train(x, y)
    probabilities = learner.batch_predict_proba(x.reshape(3, 1, 8, 8))
    assert probabilities.shape == (3, 2)
    torch.testing.assert_close(probabilities.sum(1), torch.ones(3))
