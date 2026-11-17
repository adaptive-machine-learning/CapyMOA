import pytest

pytestmark = pytest.markskip("torch")

import torch

from capymoa.core.torch import FeatureModel, ProjectionModel


def test_feature_model_requires_features():
    class Bad(FeatureModel):
        def forward(self, x):
            return x

    with pytest.raises(TypeError):
        Bad()


def test_projection_model_requires_project():
    class Half(ProjectionModel):
        def features(self, x):
            return x

        def forward(self, x):
            return x

    with pytest.raises(TypeError):
        Half()

    class Full(Half):
        def project(self, features):
            return features * 2

    assert isinstance(Full(), FeatureModel)
    assert Full().project(torch.ones(1)).item() == 2
