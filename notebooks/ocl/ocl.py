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
# Continual learning trains a model on a sequence of tasks and asks it to perform well
# on all tasks seen so far. A task represents a concept or relation to learn. In online
# continual learning (OCL), the model sees each example once and predicts throughout the
# stream.
#
# A typical data-stream learner adapts to changes in the data distribution without
# needing to retain past knowledge. Continual learning aims to preserve performance on
# earlier tasks while learning new ones. In deep learning, losing that past performance
# is called *catastrophic forgetting*; some non-deep models may be less prone to it.
#
# This notebook compares some of CapyMOA's [built-in OCL
# strategies](https://capymoa.org/api/modules/capymoa.ocl.strategy.html) on a small
# dataset. The hyper-parameters were tuned in advance. To add your own learner, see the
# [custom learner](https://capymoa.org/notebooks/ocl/custom.html) notebook.

# %% tags=["remove-cell"]
# This cell is hidden on capymoa.org. See docs/contributing/docs.rst
from capymoa._nbmock import is_nb_fast, mock_datasets

if is_nb_fast():
    mock_datasets()

# %% [markdown]
# ## Dataset
#
# {class}`~capymoa.ocl.datasets.TinySplitMNIST` has 16x16 images of ten digits. It has
# 100 training and 20 test images per digit. The digits are split into five tasks of two
# classes each. The learner sees the tasks one after the other, and is tested on all
# digits (class-incremental setting).

# %%
import torch

from capymoa.ocl import evaluate_ocl
from capymoa.ocl.datasets import TinySplitMNIST

# These models are tiny. One thread is faster than many.
torch.set_num_threads(1)

scenario = TinySplitMNIST()
print(scenario.task_schedule)

# %% [markdown]
# ## Model
#
# In this demo all strategies share the built-in
# {class}`~capymoa.core.torch.ann.Perceptron`. It exposes hidden features for strategies
# that need them.


# %%
from capymoa.core.torch.ann import Perceptron


def make_model() -> Perceptron:
    return Perceptron(scenario.schema, hidden_size=128)


# %% [markdown]
# ## Strategies
#
# We compare the following. All replay strategies keep 200 examples.
#
# * {class}`~capymoa.classifier.Finetune` trains on each batch and has no defence against forgetting.
# * {class}`~capymoa.ocl.strategy.ExperienceReplay` mixes past examples into each batch.
# * {class}`~capymoa.ocl.strategy.DERPP` replays past examples with their old logits (Buzzega et al., 2020).
# * {class}`~capymoa.ocl.strategy.PCR` trains the model with a contrastive loss on class proxies (Lin et al., 2023).
# * {class}`~capymoa.ocl.strategy.RanPAC` uses a fixed random projection, with a classifier fit in closed form (McDonnell et al., 2023).
#
# Each function builds a fresh learner from a seed.

# %%
from torch import Tensor

from capymoa.classifier import Finetune
from capymoa.ocl.strategy import DERPP, PCR, ExperienceReplay, RanPAC


def augment(x: Tensor) -> Tensor:
    """Shift the batch by up to one pixel and add noise."""
    shift = torch.randint(-1, 2, (2,)).tolist()
    return torch.roll(x, shifts=shift, dims=(-2, -1)) + 0.05 * torch.randn_like(x)


def make_finetune(seed: int):
    torch.manual_seed(seed)
    model = make_model()
    optimiser = torch.optim.Adam(model.parameters(), lr=3.6e-4)
    return Finetune(scenario.schema, model, optimizer=optimiser, random_seed=seed)


def make_er(seed: int):
    torch.manual_seed(seed)
    model = make_model()
    optimiser = torch.optim.Adam(model.parameters(), lr=3.4e-3)
    learner = Finetune(scenario.schema, model, optimizer=optimiser, random_seed=seed)
    return ExperienceReplay(learner, buffer_size=200, repeat=5)


def make_derpp(seed: int):
    torch.manual_seed(seed)
    model = make_model()
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
        make_model(),
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
# We train each learner on the stream with a batch size of 10 and test on every task
# after each task. We repeat this with three seeds to see the spread. The plots below
# show the first seed.

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
            eval_window_size=100,
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
#
# The table reports mean final accuracy across tasks and seeds, its standard deviation,
# and mean anytime accuracy on tasks seen so far. Anytime accuracy is measured four
# times during each task; the seen-task measure includes only tasks learned by that
# point. Finetune forgets older tasks, while replay and RanPAC do better. The strategies
# were tuned on other seeds and the test set, so scores here may be slightly high.

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
# This plot shows the accuracy on the tasks seen so far, measured four times per task.
# Markers are the accuracy at the end of each task. The line is the mean over seeds and
# the band is one standard deviation.
#

# %%
from capymoa.ocl.plot import plot_accuracy_comparison

fig, ax = plt.subplots(figsize=(8, 4), layout="constrained")
plot_accuracy_comparison(runs, ax=ax);

# %% [markdown]
# ## Per-task accuracy
#
# Each plot shows the accuracy on each task after training on each task. A learner that
# forgets has low values in the lower left.
#

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
# A confusion matrix compares true classes (rows) with predicted classes (columns).
# Here, classes are ordered by when they first appear in the stream, and each row is
# normalized to show the fraction of that true class assigned to each prediction.
# Errors in the last columns can indicate bias toward recent tasks.

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
#
# The plot combines the task-level and anytime measures above with two standard online
# stream metrics. **Online window accuracy** is test-then-train accuracy over a moving
# window. We set `eval_window_size=100` to show several windows over this 1,000-example
# stream; the default of 1,000 would yield only one completed window. **Online average
# accuracy** is cumulative test-then-train accuracy over the stream.

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
#
# ## OCL and Recurrent Data Streams
#
# Here we build a {class}`~capymoa.stream.drift.RecurrentConceptDriftStream` from four
# {class}`~capymoa.stream.generator.RandomRBFGenerator` concepts (see the
# [concept drift tutorial](../drift/drift_streams.ipynb)). Each concept places 50
# radial clusters in its own region of a 20-dimensional space, and each concept
# appears four times.

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
from torch import nn

from capymoa.core.torch import FeatureModel


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
# On the first pass through the four concepts, both neural learners start with low
# accuracy on each unseen concept and improve as examples arrive. When the concepts return on
# later passes, `Finetune` drops sharply at every red drift line because training on
# newer concepts overwrote the old ones. It has to relearn each concept from scratch.
# `Experience Replay` keeps past examples from all four concepts in its buffer, so its
# accuracy stays high across concept switches. The `Hoeffding Tree` does not forget, yet
# its axis-aligned splits struggle to separate radial clusters across all four concepts.
#
# ## References
#
# * Buzzega, P., Boschini, M., Porrello, A., Abati, D., & Calderara, S. (2020). Dark
#   Experience for General Continual Learning: A Strong, Simple Baseline. Advances in
#   Neural Information Processing Systems 33, 15920–30.
#   https://papers.nips.cc/paper/2020/hash/b704ea2c39778f07c617f6b7ce480e9e-Abstract.html
# * Gunasekara, N., Pfahringer, B., Gomes, H. M., Bifet, A., & Koh, Y. S. (2024).
#   Recurrent concept drifts on data streams. International Joint Conferences on
#   Artificial Intelligence Organization.
# * Lin, H., Zhang, B., Feng, S., Li, X., & Ye, Y. (2023). PCR: Proxy-Based Contrastive
#   Replay for Online Class-Incremental Continual Learning. CVPR, 24246–24255.
#   https://openaccess.thecvf.com/content/CVPR2023/html/Lin_PCR_Proxy-Based_Contrastive_Replay_for_Online_Class-Incremental_Continual_Learning_CVPR_2023_paper.html
# * McDonnell, M. D., Gong, D., Parvaneh, A., Abbasnejad, E., & van den Hengel, A.
#   (2023). RanPAC: Random Projections and Pre-trained Models for Continual Learning.
#   NeurIPS 2023. https://arxiv.org/abs/2307.02251
# * Rolnick, D., Ahuja, A., Schwarz, J., Lillicrap, T., & Wayne, A. (2019). Experience
#   replay for continual learning. Advances in neural information processing systems,
#   32. https://arxiv.org/abs/1811.11682
