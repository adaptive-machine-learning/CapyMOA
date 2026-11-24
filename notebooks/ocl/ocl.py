# ---
# jupyter:
#   jupytext:
#     default_lexer: ipython3
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.5
#   kernelspec:
#     display_name: .venv
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Online continual learning
#
# In online continual learning (OCL), a model learns from a stream of tasks. It sees each example once, must predict at any time, and must not forget old tasks while it learns new ones. Forgetting old tasks is called *catastrophic forgetting*.
#
# This notebook runs some of CapyMOA's built-in OCL strategies on a small dataset that is fast on a CPU. The hyper-parameters were tuned in advance. To add your own learner, see the [custom learner](custom.ipynb) notebook.
#
# *More information about CapyMOA can be found at* https://www.capymoa.org.
#
# **last updated on 24/11/2026**

# %% tags=["remove-cell"]
# This cell is hidden on capymoa.org. See docs/contributing/docs.rst
from capymoa._nbmock import is_nb_fast, mock_datasets

if is_nb_fast():
    mock_datasets()

# %% [markdown]
# ## Dataset
#
# {class}`~capymoa.ocl.datasets.TinySplitMNIST` has 16x16 images of ten digits. It has 100 training and 20 test images per digit. The digits are split into five tasks of two classes each. The learner sees the tasks one after the other, and is tested on all digits (class-incremental setting).

# %%
import torch
from torch import Tensor, nn

from capymoa.core.torch import FeatureModel
from capymoa.ocl import evaluate_ocl
from capymoa.ocl.datasets import TinySplitMNIST

# These models are tiny. One thread is faster than many.
torch.set_num_threads(1)

scenario = TinySplitMNIST()
print(scenario.task_schedule)

# %% [markdown]
# ## Model
#
# All neural strategies share one small multi-layer perceptron. It is a {class}`~capymoa.core.torch.FeatureModel`, since some strategies need the hidden features.


# %%
class MLP(FeatureModel):
    def __init__(self, hidden_size: int = 128):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Flatten(),
            nn.Linear(scenario.schema.get_num_attributes(), hidden_size),
            nn.ReLU(),
        )
        self.head = nn.Linear(hidden_size, scenario.schema.get_num_classes())

    def features(self, x: Tensor) -> Tensor:
        return self.encoder(x)

    def forward(self, x: Tensor) -> Tensor:
        return self.head(self.features(x))


# %% [markdown]
# ## Strategies
#
# We compare the following. All replay strategies keep 200 examples.
#
# * **Finetune** trains on each batch and has no defence against forgetting.
# * **Experience Replay (ER)** mixes a random sample of past examples into each batch.
# * **DER++** replays past examples with their old logits (Buzzega et al., 2020).
# * **PCR** trains the model with a contrastive loss on class proxies (Lin et al., 2023).
# * **RanPAC** uses a fixed random projection, with a classifier fit in closed form (McDonnell et al., 2023).
#
# Each function builds a fresh learner from a seed.

# %%
from capymoa.classifier import Finetune
from capymoa.ocl.strategy import PCR, DERPP, ExperienceReplay, RanPAC


def augment(x: Tensor) -> Tensor:
    """Shift the batch by up to one pixel and add noise."""
    shift = torch.randint(-1, 2, (2,)).tolist()
    return torch.roll(x, shifts=shift, dims=(-2, -1)) + 0.05 * torch.randn_like(x)


def make_finetune(seed: int):
    torch.manual_seed(seed)
    model = MLP()
    optimiser = torch.optim.Adam(model.parameters(), lr=3.6e-4)
    return Finetune(scenario.schema, model, optimizer=optimiser, random_seed=seed)


def make_er(seed: int):
    torch.manual_seed(seed)
    model = MLP()
    optimiser = torch.optim.Adam(model.parameters(), lr=3.4e-3)
    learner = Finetune(scenario.schema, model, optimizer=optimiser, random_seed=seed)
    return ExperienceReplay(learner, buffer_size=200, repeat=5)


def make_derpp(seed: int):
    torch.manual_seed(seed)
    model = MLP()
    optimiser = torch.optim.Adam(model.parameters(), lr=4.1e-3)
    return DERPP(
        scenario.schema,
        model,
        optimiser,
        alpha=1.33,
        beta=2.38,
        substeps=4,
        buffer_capacity=200,
        seed=seed,
    )


def make_pcr(seed: int):
    torch.manual_seed(seed)
    return PCR(
        scenario.schema,
        MLP(),
        feature_dim=128,
        augment=augment,
        optimiser_class=torch.optim.Adam,
        optimiser_kwargs={"lr": 3.6e-3},
        temperature=0.117,
        buffer_capacity=200,
        replay_batch_size=20,
        substeps=4,
        seed=seed,
    )


def make_ranpac(seed: int):
    return RanPAC(scenario.schema, projection_dim=2048, ridge=2.28, random_seed=seed)


strategies = {
    "Finetune": make_finetune,
    "ER": make_er,
    "DER++": make_derpp,
    "PCR": make_pcr,
    "RanPAC": make_ranpac,
}

# %% [markdown]
# ## Evaluation
#
# We train each learner on the stream with a batch size of 10 and test on every task after each task. We repeat this with three seeds to see the spread. The plots below show the first seed.

# %%
import pandas as pd

seeds = (10,) if is_nb_fast() else (10, 11, 12)
runs = {
    name: [
        evaluate_ocl(
            make(seed),
            scenario.train_loaders(10),
            scenario.test_loaders(10),
            continual_evaluations=4,
        )
        for seed in seeds
    ]
    for name, make in strategies.items()
}
results = {name: r[0] for name, r in runs.items()}


def collect(key: str) -> pd.DataFrame:
    """Metric `key` for each strategy (row) and seed (column)."""
    return pd.DataFrame({name: [r[key] for r in rs] for name, rs in runs.items()}).T


final = collect("accuracy_final")
summary = pd.DataFrame(
    {
        "Final accuracy": final.mean(axis=1),
        "Std.": final.std(axis=1),
        "Anytime accuracy (seen)": collect("anytime_accuracy_seen_avg").mean(axis=1),
    }
)
summary.round(3)

# %% [markdown]
# The table shows the mean final accuracy on all tasks, its standard deviation over seeds, and the mean anytime accuracy on seen tasks (explained below). Finetune forgets older tasks. The replay strategies and RanPAC do much better. Each strategy was tuned on other seeds, and the test set, so its score here can be slightly high.

# %%
ax = summary["Final accuracy"].plot.bar(yerr=summary["Std."], rot=0, capsize=4)
ax.set_ylabel("Final accuracy")
ax.set_title("TinySplitMNIST");

# %% [markdown]
# ## Accuracy over tasks
#
# This plot shows the accuracy on the tasks seen so far, after training on each task. The line is the mean over seeds and the band is one standard deviation.

# %%
from matplotlib import pyplot as plt

from capymoa.ocl.plot import plot_accuracy_comparison

plot_accuracy_comparison(runs);

# %% [markdown]
# ## Per-task accuracy
#
# Each plot shows the accuracy on each task after training on each task. A learner that forgets has low values in the lower left.

# %%
from capymoa.ocl.plot import plot_accuracy_matrix

fig, axes = plt.subplots(
    1, len(strategies), figsize=(5.5 * len(strategies), 4.5), layout="constrained"
)
for ax, (name, r) in zip(axes, results.items()):
    plot_accuracy_matrix(r, ax=ax)
    ax.set_title(name)

# %% [markdown]
# ## Confusion matrix
#
# Classes are in the order they appear in the stream. Each row shows how often a true class gets each prediction, from 0% to 100%. Errors in the last columns mean the learner is biased to recent tasks.

# %%
from capymoa.ocl.plot import plot_confusion_matrix

fig, axes = plt.subplots(
    1, len(strategies), figsize=(5.5 * len(strategies), 4.5), layout="constrained"
)
for ax, (name, r) in zip(axes, results.items()):
    plot_confusion_matrix(r, ax=ax)
    ax.set_title(name)
fig.colorbar(ax.images[0], ax=axes, label="Fraction of true class");

# %% [markdown]
# ## Accuracy during training
#
# These are useful measures of accuracy for a continual learner.
#
# * **Acc. (all)**
#
#   This is the accuracy on all tasks including unseen ones after training completes on each task.
#
# * **Acc. (seen)**
#
#   This is the accuracy only on seen tasks after completing training on each task.
#
# * **Anytime Acc. (all/seen)**
#
#   Like "Acc. (all/seen)" but measured periodically during the training of each task.
#
# * **Avg. Anytime Acc. (all/seen)**
#
#   This is the average of "Anytime Acc. (all/seen)".
#
# * **Online Win. Acc.**
#
#   This is the usual data stream windowed test-then-train accuracy over the data stream.
#
# * **Online Avg. Acc.**
#
#   This is the usual cumulative test-then-train accuracy over the data stream.

# %%
from capymoa.ocl.plot import plot_accuracy

fig, axes = plt.subplots(
    1, len(strategies), figsize=(5.5 * len(strategies), 4.5), layout="constrained"
)
for ax, (name, r) in zip(axes, results.items()):
    plot_accuracy(r, ax=ax)
    ax.set_title(name)

# %% [markdown]
# Each result is a dictionary. Besides scalars, it has `per_task` and `anytime` column dictionaries (use `pd.DataFrame(...)` for a table), arrays like `accuracy_matrix`, and `ttt` (test-then-train), the usual online evaluation of a data stream classifier.
#
# ## References
#
# * Buzzega, P., Boschini, M., Porrello, A., Abati, D., & Calderara, S. (2020). Dark Experience for General Continual Learning. NeurIPS.
# * Lin, H., Zhang, B., Feng, S., Li, X., & Ye, Y. (2023). PCR: Proxy-based Contrastive Replay for Online Class-Incremental Continual Learning. CVPR.
# * McDonnell, M. D., Gong, D., Parveneh, A., Abbasnejad, E., & van den Hengel, A. (2023). RanPAC: Random Projections and Pre-trained Models for Continual Learning. NeurIPS.
