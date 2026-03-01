"""Learning to Prompt

* https://github.com/google-research/l2p (JAX original)
* https://github.com/JH-LEE-KR/l2p-pytorch (PyTorch reimplementation)
* https://github.com/ContinualAI/avalanche/blob/master/avalanche/models/prompt.py
* https://github.com/Christina200/Online-LoRA-official/blob/main/Si-blurry/models/l2p.py

"""

from capymoa.base import BatchClassifier

from abc import ABC, abstractmethod
from typing import Any, Optional, Tuple, Callable

from capymoa.stream import Schema
from capymoa.ocl.base import TrainTaskAware
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class L2PViT(ABC):
    """Abstract interface for vision transformer backbones used in L2P."""

    @abstractmethod
    def get_embedding_size(self) -> int:
        """Get the dimension of the patch embeddings.

        :return: The embedding dimension.
        """

    @abstractmethod
    def get_patch_embed(self, pixel_values: Tensor) -> Tensor:
        """Turn pixel values into patch embeddings.

        :param pixel_values: A tensor of shape (batch_size, channels, height, width)
        :return: A tensor of shape (batch_size, num_patches + 1, embed_dim)
        """

    @abstractmethod
    def forward_encoder(self, prompts: Tensor, patch_embed: Tensor) -> Tensor:
        """Encode the patch embeddings with the given prompts.

        :param prompts: A tensor of shape (batch_size, prompt_length, embed_dim)
        :param patch_embed: A tensor of shape (batch_size, num_patches + 1, embed_dim)
        :return: A tensor of shape (batch_size, num_patches + 1 + prompt_length, embed_dim)
        """

    @abstractmethod
    def forward_query(self, patch_embed: Tensor) -> Tensor:
        """Get the encoded query embedding from the patch embeddings.

        :param patch_embed: A tensor of shape (batch_size, num_patches + 1, embed_dim)
        :return: A tensor of shape (batch_size, embed_dim)
        """


class _HuggingFaceAdapter(L2PViT, nn.Module):
    def __init__(self, pretrained_name: str) -> None:
        super().__init__()
        try:
            from transformers import AutoModel, AutoImageProcessor
        except ImportError:
            raise ImportError(
                "Transformers is not installed. Please install it with `pip install transformers`."
            )
        self._model = AutoModel.from_pretrained(pretrained_name)
        self._processor = AutoImageProcessor.from_pretrained(
            pretrained_name, use_fast=True
        )

    def get_embedding_size(self) -> int:
        return self._model.config.hidden_size

    def get_patch_embed(self, pixels: Tensor) -> Tensor:
        pixels = self._processor(
            images=pixels, return_tensors="pt", do_rescale=False
        ).pixel_values
        return self._model.embeddings(pixels)

    def forward_encoder(self, prompts: Tensor, patch_embed: Tensor) -> Tensor:
        cls_token = patch_embed[:, :1, :]
        other_patches = patch_embed[:, 1:, :]

        # Add CLS position embedding to prompts
        cls_pos_embedding = self._model.embeddings.position_embeddings[:, :1]
        prompts = prompts + cls_pos_embedding

        embedding = torch.cat([cls_token, prompts, other_patches], dim=1)
        sequence_output = self._model.encoder(embedding).last_hidden_state
        sequence_output = self._model.layernorm(sequence_output)
        return sequence_output

    @torch.no_grad()
    def forward_query(self, patch_embedding: Tensor) -> Tensor:
        # Return [CLS] token embedding
        sequence_output = self._model.encoder(patch_embedding).last_hidden_state
        sequence_output = self._model.layernorm(sequence_output)
        return sequence_output[:, 0, :]


def _prompt_lookup(
    query: Tensor, keys: Tensor, prompts: Tensor, top_k: int
) -> Tuple[Tensor, Tensor]:
    """Find prompts with keys closest to the query.

    :param query: Query of shape (batch_size, embedding_dimension)
    :param keys: Keys of shape (pool_size, embedding_dimension)
    :param prompts: Prompts of shape (pool_size, prompt_length, embedding_dimension)
    :param top_k: Number of prompts to select
    :return: Tuple containing:
        - Selected prompts of shape (batch_size, top_k * prompt_length, embedding_dimension)
        - Average cosine distance between selected keys and query.
    """
    batch_size, embedding_dimension = query.shape
    pool_size, prompt_length, _ = prompts.shape
    assert query.shape == (batch_size, embedding_dimension)
    assert keys.shape == (pool_size, embedding_dimension)
    assert prompts.shape == (pool_size, prompt_length, embedding_dimension)
    assert top_k <= pool_size

    cosine_distance = 1 - F.cosine_similarity(query.unsqueeze(1), keys, dim=-1)
    _, idx = cosine_distance.topk(top_k, dim=1, largest=False)

    selected = prompts[idx].view(batch_size, top_k * prompt_length, embedding_dimension)
    return selected, cosine_distance[:, idx].mean()


class _PromptPool(nn.Module):
    def __init__(
        self,
        pool_size: int,
        prompt_length: int,
        embed_dim: int,
        top_k: int,
        num_tasks: int = 1,
    ) -> None:
        """Learning 2 Prompt Pool module.

        :param pool_size: Number of prompts in the pool
        :param prompt_length: Length of each prompt (number of tokens/patches)
        :param embed_dim: Dimension of the embeddings
        :param top_k: Number of top prompts to retrieve per query
        :param num_tasks: Number of tasks for task-specific prompts
        """
        super().__init__()
        self.pool_size = pool_size
        self.prompt_length = prompt_length
        self.embed_dim = embed_dim
        self.top_k = top_k
        self.num_tasks = num_tasks
        self.prompts_per_task = pool_size // num_tasks

        self.keys = nn.Parameter(torch.empty(pool_size, embed_dim))
        self.prompts = nn.Parameter(torch.empty(pool_size, prompt_length, embed_dim))

        nn.init.uniform_(self.keys, -1, 1)
        nn.init.uniform_(self.prompts, -1, 1)

    def forward(
        self,
        query: Tensor,
        task_id: Optional[int] = None,
    ) -> Tuple[Tensor, Tensor]:
        if task_id is not None:
            start = task_id * self.prompts_per_task
            end = (task_id + 1) * self.prompts_per_task
            return _prompt_lookup(
                query=query,
                keys=self.keys[start:end],
                prompts=self.prompts[start:end],
                top_k=self.top_k,
            )

        return _prompt_lookup(
            query=query,
            keys=self.keys,
            prompts=self.prompts,
            top_k=self.top_k,
        )


class _L2PModel(nn.Module):
    def __init__(self, vit: L2PViT, prompt_pool: _PromptPool, out_features: int):
        super().__init__()
        if isinstance(vit, nn.Module):
            vit = vit.eval().requires_grad_(False)
        else:
            raise ValueError("vit must be an instance of nn.Module.")

        self.vit = vit
        self.prompt_pool = prompt_pool
        self.head = nn.Linear(vit.get_embedding_size(), out_features)

    def forward(
        self, x: Tensor, task_id: Optional[int] = None
    ) -> Tuple[Tensor, Tensor]:
        # First forward pass to get query
        patch_embed = self.vit.get_patch_embed(x)
        query = self.vit.forward_query(patch_embed)
        prompts, cosine_distance = self.prompt_pool.forward(query, task_id=task_id)

        # Second forward pass this time with prompts
        encoded_patches = self.vit.forward_encoder(prompts, patch_embed)
        prompt_out_len = self.prompt_pool.top_k * self.prompt_pool.prompt_length

        # Trim [CLS] token and average prompt outputs
        features = encoded_patches[:, 1 : 1 + prompt_out_len].mean(dim=1)
        return self.head(features), cosine_distance


class L2P(BatchClassifier, TrainTaskAware):
    """Learning to Prompt.

    Learning to Prompt (L2P) [#f1]_ is a continual learning strategy that leverages a
    pool of learnable prompts to adapt a pre-trained vision transformer (ViT) to new
    tasks. For each input, the most relevant prompts are selected from the pool based on
    the similarity between the input's embedding and the prompt keys. The selected
    prompts are then used to condition the ViT, allowing it to effectively learn new
    tasks while mitigating catastrophic forgetting.

    L2P relies on knowledge of the tasks during training to select task-specific prompts
    but does not require task information during inference.

    ..  code-block:: python

        # Please note this code block is not regularly tested.
        from capymoa.ocl.strategy.l2p import L2P
        from capymoa.ocl.datasets import SplitCIFAR100
        from capymoa.ocl.evaluation import ocl_train_eval_loop
        scenario = SplitCIFAR100()
        learner = L2P(scenario.schema, scenario.task_mask, device="cuda")
        results = ocl_train_eval_loop(
            learner,
            scenario.train_loaders(32),
            scenario.test_loaders(32),
            progress_bar=True
        )
        print(f"{results.accuracy_final*100:.1f}%")

    ..  [#f1] Wang, Z., Zhang, Z., Lee, C.-Y., Zhang, H., Sun, R., Ren, X., Su, G.,
        Perot, V., Dy, J. G., & Pfister, T. (2022). Learning to prompt for continual
        learning. IEEE/CVF Conference on Computer Vision and Pattern Recognition, CVPR
        2022, New Orleans, LA, USA, June 18-24, 2022, 139-149.
        https://doi.org/10.1109/CVPR52688.2022.00024
    """

    def __init__(
        self,
        schema: Schema,
        task_mask: Tensor,
        vit: L2PViT | str = "facebook/dinov2-small",
        prompts_per_task: int = 5,
        prompt_length: int = 1,
        top_k: int = 3,
        pull_constraint_coeff: float = 0.1,
        optimizer: Callable[[Any], torch.optim.Optimizer] = lambda params: (
            torch.optim.Adam(params, lr=0.01)
        ),
        device: str = "cpu",
        random_seed: int = 1,
    ):
        """Construct L2P learner.

        :param schema: Schema describing the datastream.
        :param task_mask: A boolean tensor of shape (num_tasks, num_classes) indicating
            which classes belong to each task.
        :param vit: Vision transformer backbone or the name of a pretrained model from
            HuggingFace Transformers. Requires `transformers` to be installed.
        :param prompts_per_task: Number of prompts per task in the prompt pool.
        :param prompt_length: Length of each prompt (number of tokens/patches).
        :param top_k: Number of top prompts to retrieve per query.
        :param pull_constraint_coeff: Coefficient for the pull constraint loss term.
        :param optimizer: Function that takes model parameters and returns an optimizer instance.
        :param device: Device to run the model on, e.g., "cpu" or "cuda".
        :param random_seed: Random seed for reproducibility.
        :param logger: Optional logger for tracking training metrics.
        """
        super().__init__(schema, random_seed)
        if isinstance(vit, str):
            vit = _HuggingFaceAdapter(vit)

        self.device = torch.device(device)
        self._num_tasks = task_mask.size(0)
        self._pull_constraint_coeff = pull_constraint_coeff
        self._model = _L2PModel(
            vit=vit,
            prompt_pool=_PromptPool(
                pool_size=prompts_per_task * self._num_tasks,
                prompt_length=prompt_length,
                embed_dim=vit.get_embedding_size(),
                top_k=top_k,
                num_tasks=self._num_tasks,
            ),
            out_features=schema.get_num_classes(),
        ).to(self.device)
        self._new_optimizer_fn = optimizer
        self._optimizer = self._new_optimizer()
        self._mask = task_mask.to(self.device)
        self._train_task = 0

    def _new_optimizer(self) -> torch.optim.Optimizer:
        return self._new_optimizer_fn(
            (p for p in self._model.parameters() if p.requires_grad)
        )

    def on_train_task(self, task_id: int):
        self._train_task = task_id
        self._optimizer = self._new_optimizer()

    def batch_train(self, x: torch.Tensor, y: torch.Tensor) -> None:
        x = x.view(x.size(0), *self.schema.shape)  # Reshape to the original shape
        self._optimizer.zero_grad()
        y_hat, cosine_distance = self._model(x, self._train_task)
        y_hat = self._mask[self._train_task] * y_hat
        cosine_distance = cosine_distance * self._pull_constraint_coeff
        ce_loss = nn.functional.cross_entropy(y_hat, y)

        loss = ce_loss + cosine_distance

        acc = (y_hat.argmax(dim=1) == y).float().mean().item()
        print(
            f"loss: {loss.item():.2f} "
            f"ce: {ce_loss.item():.2f} "
            f"pull: {cosine_distance.item():.2f} "
            f"acc: {acc * 100:.1f}%",
            end="\r",
        )

        loss.backward()
        self._optimizer.step()

    def batch_predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        x = x.view(x.size(0), *self.schema.shape)  # Reshape to the original shape
        y_hat, _ = self._model(x)
        return F.softmax(y_hat, dim=1)
