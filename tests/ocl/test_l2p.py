"""Tests for L2P strategy.

Unfortunantely we cannot test with the real ViT model due to resource limits but we
can mock a dummy ViT model to test that the training loop does not error at least.
"""

from capymoa.ocl.strategy.l2p import L2P, L2PViT
from capymoa.ocl.datasets import TinySplitMNIST
from capymoa.ocl.evaluation import ocl_train_eval_loop
from torch import Tensor, nn
import torch


class DummyViT(L2PViT, nn.Module):
    seq_len = 4
    emebed_dim = 8

    def get_embedding_size(self) -> int:
        return self.emebed_dim

    def get_patch_embed(self, pixel_values: Tensor) -> Tensor:
        return torch.randn((pixel_values.size(0), self.seq_len, self.emebed_dim))

    def forward_encoder(self, prompts: Tensor, patch_embed: Tensor) -> Tensor:
        return torch.randn((patch_embed.size(0), self.seq_len, self.emebed_dim))

    def forward_query(self, patch_embed: Tensor) -> Tensor:
        return torch.randn((patch_embed.size(0), self.emebed_dim))


def test_l2p():
    scenario = TinySplitMNIST()
    learner = L2P(
        scenario.schema,
        scenario.task_mask,
        vit=DummyViT(),
        device="cpu",
    )
    ocl_train_eval_loop(
        learner,
        scenario.train_loaders(32),
        scenario.test_loaders(32),
    )
