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
from matplotlib import pyplot as plt

fig, ax = plt.subplots(figsize=(7, 4), layout="constrained")
summary["Final accuracy"].plot.bar(yerr=summary["Std."], rot=0, capsize=4, ax=ax)
ax.set_ylabel("Final accuracy")
ax.set_ylim(0, 1)
ax.grid(axis="y", alpha=0.25)
ax.set_title("TinySplitMNIST");

# %% [markdown]
# ## Accuracy over tasks
#
# This plot shows the accuracy on the tasks seen so far, measured four times per task. Markers are the accuracy at the end of each task. The line is the mean over seeds and the band is one standard deviation.

# %%
from capymoa.ocl.plot import plot_accuracy_comparison

fig, ax = plt.subplots(figsize=(8, 4), layout="constrained")
plot_accuracy_comparison(runs, ax=ax);

# %% [markdown]
# ## Per-task accuracy
#
# Each plot shows the accuracy on each task after training on each task. A learner that forgets has low values in the lower left.

# %%
from capymoa.ocl.plot import plot_accuracy_matrix

fig, axes = plt.subplots(
    2, 3, figsize=(13.5, 8), layout="constrained", squeeze=False
)
for ax, (name, r) in zip(axes.flat, results.items(), strict=False):
    plot_accuracy_matrix(r, ax=ax)
    ax.set_title(name)
axes.flat[-1].set_visible(False)

# %% [markdown]
# ## Confusion matrix
#
# Classes are in the order they appear in the stream. Each row shows how often a true class gets each prediction, from 0% to 100%. Errors in the last columns mean the learner is biased to recent tasks.

# %%
from capymoa.ocl.plot import plot_confusion_matrix

fig, axes = plt.subplots(
    2, 3, figsize=(13.5, 8), layout="constrained", squeeze=False
)
for ax, (name, r) in zip(axes.flat, results.items(), strict=False):
    plot_confusion_matrix(r, ax=ax)
    ax.set_title(name)
axes.flat[-1].set_visible(False)
fig.colorbar(
    axes.flat[4].images[0],
    ax=axes.ravel()[:5].tolist(),
    label="Fraction of true class",
);

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
    2, 3, figsize=(13.5, 8.5), layout="constrained", squeeze=False
)
handles = None
labels = None
for ax, (name, r) in zip(axes.flat, results.items(), strict=False):
    plot_accuracy(r, ax=ax)
    ax.set_title(name)
    if handles is None:
        handles, labels = ax.get_legend_handles_labels()
    legend = ax.get_legend()
    if legend is not None:
        legend.remove()
axes.flat[-1].set_visible(False)
assert handles is not None and labels is not None
fig.legend(
    handles,
    labels,
    loc="outside lower center",
    ncol=3,
    frameon=False,
)

# %% [markdown]
# Each result is a dictionary. Besides scalars, it has `per_task` and `anytime` column dictionaries (use `pd.DataFrame(...)` for a table), arrays like `accuracy_matrix`, and `ttt` (test-then-train), the usual online evaluation of a data stream classifier.
#
# ## OCL and Recurrent Data Streams
#
# CapyMOA's OCL strategies are standard classifiers, so they also work with {func}`~capymoa.classifier.evaluate_classifier` on ordinary data streams. Continual learning helps when concepts return over time and occupy different parts of the input space. (If two concepts give different labels to the same inputs, replaying old examples conflicts with the active concept.)
#
# Here we build a {class}`~capymoa.stream.drift.RecurrentConceptDriftStream` from four {class}`~capymoa.stream.generator.RandomRBFGenerator` concepts (see the [concept drift tutorial](../drift/drift_streams.ipynb)). Each concept places 50 radial clusters in its own region of a 20-dimensional space, and each concept appears four times.

# %%
from typing import cast

from capymoa.classifier import HoeffdingTree, evaluate_classifier
from capymoa.evaluation.plot import plot_windowed_results
from capymoa.evaluation.results import RunInfo
from capymoa.stream import Stream
from capymoa.stream.drift import GradualDrift, RecurrentConceptDriftStream
from capymoa.stream.generator import RandomRBFGenerator

concept_list: list[Stream] = [
    RandomRBFGenerator(
        model_random_seed=seed,
        number_of_classes=10,
        number_of_attributes=20,
        number_of_centroids=50,
    )
    for seed in (1, 2, 3, 4)
]
concept_name_list = ["Concept 1", "Concept 2", "Concept 3", "Concept 4"]

concept_length = 40 if is_nb_fast() else 1000
window_size = 20 if is_nb_fast() else 250
concept_transition_width = 10 if is_nb_fast() else 50
max_recurrences_per_concept = 2 if is_nb_fast() else 4
max_instances = concept_length * len(concept_list) * max_recurrences_per_concept

stream_with_recurrent_concepts = RecurrentConceptDriftStream(
    concept_list=concept_list,
    max_recurrences_per_concept=max_recurrences_per_concept,
    transition_type_template=GradualDrift(
        position=concept_length, width=concept_transition_width
    ),
    concept_name_list=concept_name_list,
)
rbf_schema = stream_with_recurrent_concepts.get_schema()
print(stream_with_recurrent_concepts.describe(max_instances))

# %% [markdown]
# We compare a {class}`~capymoa.classifier.HoeffdingTree`, `Finetune`, and {class}`~capymoa.ocl.strategy.ExperienceReplay`. All three run in one test-then-train pass with `batch_size=1`, so each learner predicts and takes one update step per example. Using four evaluation windows per concept shows what happens right after each drift.


# %%
class RBFMLP(FeatureModel):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(rbf_schema.get_num_attributes(), 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
        )
        self.head = nn.Linear(64, rbf_schema.get_num_classes())

    def features(self, x: Tensor) -> Tensor:
        return self.encoder(x)

    def forward(self, x: Tensor) -> Tensor:
        return self.head(self.features(x))


def make_rbf_finetune(seed: int):
    torch.manual_seed(seed)
    model = RBFMLP()
    optimiser = torch.optim.SGD(model.parameters(), lr=0.05)
    return Finetune(rbf_schema, model, optimizer=optimiser, random_seed=seed)


def make_rbf_er(seed: int):
    return ExperienceReplay(make_rbf_finetune(seed), buffer_size=800, repeat=1)


recurrent_results = evaluate_classifier(
    stream_with_recurrent_concepts,
    {
        "Hoeffding Tree": HoeffdingTree(schema=rbf_schema, random_seed=10),
        "Finetune": make_rbf_finetune(seed=10),
        "Experience Replay": make_rbf_er(seed=10),
    },
    window_size=window_size,
    max_instances=max_instances,
    batch_size=1,
)
plot_windowed_results(
    cast(RunInfo, recurrent_results["Hoeffding Tree"]),
    cast(RunInfo, recurrent_results["Finetune"]),
    cast(RunInfo, recurrent_results["Experience Replay"]),
    metric="accuracy",
    plot_title="Recurrent RBF concepts",
    ylabel="Accuracy",
    save_only=False,
)
pd.Series(
    {
        name: result["cumulative"]["accuracy"]
        for name, result in recurrent_results.items()
    },
    name="Cumulative accuracy",
).to_frame().round(1)

# %% [markdown]
# On the first pass through the four concepts, both neural learners start low on each unseen concept and improve as examples arrive. When the concepts return on later passes, `Finetune` drops sharply at every red drift line because training on newer concepts overwrote the old ones. It has to relearn each concept from scratch. `Experience Replay` keeps past examples from all four concepts in its buffer, so its accuracy stays high across concept switches. The `Hoeffding Tree` does not forget, yet its axis-aligned splits struggle to separate radial clusters across all four concepts.
#
# ## References
#
# * Buzzega, P., Boschini, M., Porrello, A., Abati, D., & Calderara, S. (2020). Dark Experience for General Continual Learning. NeurIPS.
# * Gunasekara, N., Pfahringer, B., Gomes, H. M., Bifet, A., & Koh, Y. S. (2024). Recurrent Concept Drifts on Data Streams. IJCAI.
# * Lin, H., Zhang, B., Feng, S., Li, X., & Ye, Y. (2023). PCR: Proxy-based Contrastive Replay for Online Class-Incremental Continual Learning. CVPR.
# * McDonnell, M. D., Gong, D., Parveneh, A., Abbasnejad, E., & van den Hengel, A. (2023). RanPAC: Random Projections and Pre-trained Models for Continual Learning. NeurIPS.
# * Rolnick, D., Ahuja, A., Schwarz, J., Lillicrap, T., & Wayne, G. (2019). Experience Replay for Continual Learning. NeurIPS.
