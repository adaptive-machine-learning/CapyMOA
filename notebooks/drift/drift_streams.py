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
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Simulating concept drifts with the DriftStream API
#
# This tutorial demonstrates how to use the DriftStream API in CapyMOA:
#
# * We start by showing how one can use a synthetic generator without concept drift (RandomTreeGenerator).
# * We delve into the two ways of defining a DriftStream:
#     * **DriftStream Position**: **`drift position` + `drift width`**.
#     * **DriftStream Range**: **`concept num_instances` + `drift num_instances`**.
# * Other examples can be found in [Exploring Advanced Features](https://capymoa.org/notebooks/common/advanced_API.html), such as configuring and manipulating MOA streams directly.
# * Also, [**Tutorial**: Drift Detection](https://capymoa.org/notebooks/drift_detection.html) complements this tutorial by demonstrating the drift detection API and the various algorithms implemented in capymoa. 
#
# ---
#
# **References:**
#
# * Cerqueira, V., Gomes, H. M., Heyden, M., Pfahringer, B., & Bifet, A. (2026). *A Framework for Evaluating and Benchmarking Concept Drift Detection Methods.* ACM SIGKDD Conference on Knowledge Discovery and Data Mining.
# * Gama, J., Žliobaitė, I., Bifet, A., Pechenizkiy, M., & Bouchachia, A. (2014). *A survey on concept drift adaptation.* ACM Computing Surveys, 46(4), 1-37.
#
# ---
#
# *More information about CapyMOA can be found at* https://www.capymoa.org.
#
# **last update on 06/08/2026**

# %% [markdown]
# ## CapyMOA synthetic generators
#
# * In this example, we show how to use `RandomTreeGenerator` to generate a synthetic stream. No concept drift, just a synthetic stream.

# %% tags=["remove-cell"]
# This cell is hidden on capymoa.org. See docs/contributing/docs.rst
from capymoa._nbmock import is_nb_fast, override_prequential_evaluation

if is_nb_fast():
    override_prequential_evaluation(max_instances=1000)

# %%
from capymoa.classifier import HoeffdingTree
from capymoa.evaluation import prequential_evaluation
from capymoa.evaluation.visualization import plot_windowed_results
from capymoa.stream.generator import RandomTreeGenerator

rtg_stream = RandomTreeGenerator()

ht = HoeffdingTree(schema=rtg_stream.get_schema())

results_ht = prequential_evaluation(
    max_instances=10000, window_size=1000, stream=rtg_stream, learner=ht
)

plot_windowed_results(results_ht, metric="accuracy")

# %% [markdown]
# ## CapyMOA DriftStream builder API
#
# In CapyMOA, the **concepts** and **drifts** are clearly outlined on a list format. There are two ways of specifying a DriftStream in this list format:
#
# 1. **DriftStream Position**: **`drift position` + `drift width`**: the start and end of a concept is determined by the presence of an `AbruptDrift` or `GradualDrift` object.
#    
#     <!-- `[SEA(1), AbruptDrift(position=1000), SEA(2), GradualDrift(position=2000, width=500), SEA(3)]` -->
#
#     * **DriftStream([**
#         * <span style="color:blue;">SEA(1)</span>,
#         * <span style="color:red;">AbruptDrift(position=1000)</span>, 
#         * <span style="color:blue;">SEA(2)</span>, 
#         * <span style="color:green;">GradualDrift(<b>position</b>=2000, <b>width</b>=500)</span>, 
#         * <span style="color:blue;">SEA(3)</span>**])**
#
# * The `GradualDrift` can also be specified in terms of `start` and `end`.
#   
#     <!-- `[SEA(1), AbruptDrift(position=1000), SEA(2), GradualDrift(start=1750, end=2250), SEA(3)]` -->
#
#     * **DriftStream([**
#         * <span style="color:blue;">SEA(1)</span>, 
#         * <span style="color:red;">AbruptDrift(position=1000)</span>, 
#         * <span style="color:blue;">SEA(2)</span>, 
#         * <span style="color:green;">GradualDrift(<b>start</b>=1750, <b>end</b>=2250)</span>, 
#         * <span style="color:blue;">SEA(3)</span>**])**
#
# 2. **DriftStream Range**: **`concept num_instances` + `drift num_instances`**: the start and end of a concept is determined by the amount of instances generated for it, the same thing can be said about `GradualDrifts` which do not have a `start` or `end` but the number of instances i.e. the `width` of that drifting region (or drifting window). Notice that we can't specify a drift `position` or drift `start` and `end` when using the **Range** version because that would be confusing and error prone. The specification of the DriftStream, in this version, doesn't explicitly tells us about the locations of the drifts on the stream, so it is less error prone if we don't allow the user to use this approach mixed with the `drift position` one. Example:
#    
#     <!-- `[Concept(SEA(1), num_instances=1000), AbruptDrift(), Concept(SEA(2), num_instances=500), GradualDrift(num_instances=500), Concept(SEA(3), num_instances=500)]` -->
#     * **DriftStream([**
#         * <span style="color:blue;"><b>Concept(</b>SEA(1), num_instances=1000<b>)</b></span>, 
#         * <span style="color:red;">AbruptDrift()</span>, 
#         * <span style="color:blue;"><b>Concept(</b>SEA(2), num_instances=500<b>)</b></span>, 
#         * <span style="color:green;">GradualDrift(num_instances=500)</span>, 
#         * <span style="color:blue;"><b>Concept(</b>SEA(3), num_instances=500<b>)</b></span>**])**
#
# * Why do we need the **`Concept()`** specification in the **`num_instances`**? The **Stream** class, i.e. base class for **SEA** and other synthetic generators do not implement the concept of `max_instances` or `num_instances`. If we were to implement that, we would lose the idea of synthetic streams being unbounded. It is a design choice, whenever we want to _control_ a stream length, it is done externally to its definition. 
# * The **`DriftStream`** specification in the `position` version does not specify the total `size` of the stream, i.e. notice how the `SEA(3)` at the end is unbounded, there is no drift object signaling its end. That is intentional as the user specifying the `DriftStream` and manipulating it defines the end of the stream externally. This is true for synthetic streams and also for limiting *snapshot* streams like **electricity** and others that are read from files.

# %% [markdown]
# ### DriftStream `position` + `width`
#
# * Specifying drift location using the *first* version.
# * We can use either position + width or start + end to define GradualDrifts in this approach.
#
#   ```sh
#   GradualDrift(position=10000, width=2000)
#   ```
#   or
#   ```sh
#   GradualDrift(start=9000, end=12000)
#   ```
# * **Important**: meta-data about the specified Drifts is accessible from the stream object.
#
# ```python
# print(f'The definition of the DriftStream is accessible through the object:\n {stream_sea2drift}')
# ```
#
# * Furthermore, this meta-data is interpreted by the `plot_windowed_results` function producing plots that automatically indicates drift locations.

# %%
from capymoa.classifier import OnlineBagging
from capymoa.stream.drift import AbruptDrift, DriftStream, GradualDrift
from capymoa.stream.generator import SEA

stream_sea2drift = DriftStream(
    stream=[
        SEA(function=1),
        AbruptDrift(position=5000),
        SEA(function=3),
        GradualDrift(position=10000, width=2000),
        # GradualDrift(start=9000, end=12000),
        SEA(function=1),
    ]
)

OB = OnlineBagging(schema=stream_sea2drift.get_schema(), ensemble_size=10)

results_sea2drift_OB = prequential_evaluation(
    stream=stream_sea2drift, learner=OB, window_size=100, max_instances=15000
)

print(
    f"The definition of the DriftStream is accessible through the object:\n {stream_sea2drift}"
)
plot_windowed_results(results_sea2drift_OB, metric="accuracy")

# %% [markdown]
# ### DriftStream `range`
#
# * This version of the DriftStream builder specifies **how long each concept and drift lasts** instead of where each drift lands, i.e. `concept num_instances` and `drift num_instances`.
# * Each component contributes its own stretch of the stream, so the definition reads as a timeline: a concept runs for its length, an `AbruptDrift()` switches at the point it is reached, and a `GradualDrift(num_instances=...)` spans its own stretch centred on that point.
# * Concepts are wrapped in `Concept(stream, num_instances=...)`. The wrapper is needed because a `Stream` has no length of its own -- MOA generators are unbounded.
# * The two forms cannot be mixed. A range definition does not say where its drifts land, so a stray `position` would describe a location the rest of the definition contradicts.

# %%
from capymoa.stream.drift import AbruptDrift, Concept, DriftStream, GradualDrift
from capymoa.stream.generator import SEA

stream_range = DriftStream(
    stream=[
        Concept(SEA(function=1), num_instances=1000),
        AbruptDrift(),
        Concept(SEA(function=2), num_instances=500),
        GradualDrift(num_instances=500),
        Concept(SEA(function=3), num_instances=500),
    ]
)

# The lengths are translated into the positions they imply, so the drift
# metadata is the same as if they had been written out by hand.
for drift in stream_range.get_drifts():
    print(drift)

# %% [markdown]
# * The lengths are translated into the positions they imply, so a `range` definition is simply another way of writing a `position` one.
# * Evaluating both and plotting them side by side makes that concrete: the drift markers land in the same places, because they *are* the same drifts.

# %%
from capymoa.classifier import OnlineBagging

# The same stream written both ways.
stream_by_range = DriftStream(
    stream=[
        Concept(SEA(function=1), num_instances=5000),
        AbruptDrift(),
        Concept(SEA(function=3), num_instances=5000),
        GradualDrift(num_instances=2000),
        Concept(SEA(function=1), num_instances=3000),
    ]
)
stream_by_position = DriftStream(
    stream=[
        SEA(function=1),
        AbruptDrift(position=5000),
        SEA(function=3),
        GradualDrift(position=11000, width=2000),
        SEA(function=1),
    ]
)

print("range   :", [str(d) for d in stream_by_range.get_drifts()])
print("position:", [str(d) for d in stream_by_position.get_drifts()])
print(
    "identical drifts:",
    [str(d) for d in stream_by_range.get_drifts()]
    == [str(d) for d in stream_by_position.get_drifts()],
)

for name, stream in (("range", stream_by_range), ("position", stream_by_position)):
    learner = OnlineBagging(schema=stream.get_schema(), ensemble_size=10)
    results = prequential_evaluation(
        stream=stream, learner=learner, window_size=100, max_instances=15000
    )
    print(f"\n{name} form:")
    plot_windowed_results(results, metric="accuracy")

# %% [markdown]
# ### What a length does and does not guarantee
#
# * `num_instances` places the drifts along the stream. It does **not** ration instances between concepts.
# * Around an `AbruptDrift` it is exact: the switch is a step, so the first concept contributes precisely its length.
# * Around a `GradualDrift` the two concepts **overlap**. Both are drawn from while the transition runs, so each of them contributes its own length *plus* a share of the window.
# * The transition is confined to the window: before it the old concept is used exclusively, after it the new one. A concept is therefore drawn from over `num_instances + (width / 2)` instances on average.
#
# The drift *window* is exactly where the definition says. The *provenance* of an individual instance inside it is probabilistic, which matters if you are building a labelled benchmark.

# %%
# Ask the stream itself rather than assuming a shape: this is the probability
# that an instance at a given position comes from the *new* concept.
drift = stream_range.get_drifts()[1]  # the GradualDrift resolved above
transition = stream_range._root  # the internal node holding that drift

print(f"window: start={drift.start}  centre={drift.position}  end={drift.end}\n")
print(f"{'instance':>10}  {'P(new concept)':>15}   where")
for n, where in [
    (drift.start - 1, "before the window"),
    (drift.start, "window opens"),
    (drift.position, "centre"),
    (drift.end, "window closes"),
    (drift.end + 1, "after the window"),
]:
    print(f"{n:>10}  {transition.probability_of_new_concept(n):>15.4f}   {where}")

print("\nOutside the window the probability is exactly 0 or exactly 1,")
print("so the older concept does not reappear once the drift has finished.")

# %% [markdown]
# ### Telling exactly how many instances came from each concept
#
# * Because the concepts overlap inside a gradual window, the definition alone does not tell you how much of each you actually got.
# * `describe(horizon)` prints it as a table and `get_concept_counts(horizon)` returns the numbers. The `horizon` defaults to the length the definition implies.
# * The **`in drift`** column is the part of a concept's draws that happened inside a gradual window. That is why a concept declared as 500 is drawn from around 740 times: its own 500, plus its share of the 500-instance transition beside it.
# * Neither needs the stream to be run first. A transition decides using only its own seeded generator and counter, so the routing is replayed without generating any data.
#
# ⚠️ **With a real, finite stream this matters.** A concept must hold enough instances for its declared length *and* for the share of any adjacent gradual window it will be drawn from. Backing a `num_instances=500` concept with exactly 500 real instances is not enough if a gradual drift sits next to it. Use these counts to size the data before running the experiment.

# %%
# describe() prints the table; get_concept_counts() returns the numbers.
print(stream_range.describe())

print("\nraw counts, in the order the concepts were defined:")
print(" ", stream_range.get_concept_counts())

# Neither needs the stream to be run, and the horizon can be anything.
print("\nover the first 1000 instances only:")
print(" ", stream_range.get_concept_counts(1000))

# %% [markdown]
# ### Choosing how a gradual drift transitions
#
# * A `GradualDrift` mixes the two concepts across its window. **How** it mixes is the `transition_function`.
# * The transition is **confined to the window**: before it, instances come from the old concept only; after it, from the new one only.
# * Two are built in:
#   * `"sigmoid"` (default) -- a smooth S-curve, parametrised from the width so it completes inside the window.
#   * `"linear"` -- the probability rises at a constant rate across the window.
# * Both are deterministic and both finish where the window finishes. They differ in *shape*, not in extent.

# %%
# The same drift under each ramp. The window runs from 1000 to 1200.
for transition in ("sigmoid", "linear"):
    stream = DriftStream(
        stream=[
            Concept(SEA(function=1), num_instances=1000),
            GradualDrift(num_instances=200, transition_function=transition),
            Concept(SEA(function=3), num_instances=1000),
        ]
    )
    node = stream._root  # the internal transition node, for illustration
    drift = stream.get_drifts()[0]
    points = [
        drift.start - 1,
        drift.start,
        1050,
        drift.position,
        1150,
        drift.end,
        drift.end + 1,
    ]
    values = [f"{node.probability_of_new_concept(n):.3f}" for n in points]
    print(f"{transition:>8}  " + "  ".join(f"{n}:{v}" for n, v in zip(points, values)))

print("\nBoth are exactly 0 before the window and exactly 1 after it.")
print("Inside, the sigmoid stays flatter at the edges and turns faster in the middle.")

# %%
import matplotlib.pyplot as plt


def ramp_over_window(transition, width=200):
    """P(new concept) across a drift, sampled either side of the window."""
    stream = DriftStream(
        stream=[
            Concept(SEA(function=1), num_instances=1000),
            GradualDrift(num_instances=width, transition_function=transition),
            Concept(SEA(function=3), num_instances=1000),
        ]
    )
    node, drift = stream._root, stream.get_drifts()[0]
    xs = range(drift.start - 50, drift.end + 51)
    return drift, xs, [node.probability_of_new_concept(n) for n in xs]


plt.figure(figsize=(9, 4))
for transition, label in (("sigmoid", "sigmoid (default)"), ("linear", "linear")):
    drift, xs, ys = ramp_over_window(transition)
    plt.plot(list(xs), ys, label=label)

plt.axvline(drift.start, color="grey", ls=":", lw=1)
plt.axvline(drift.end, color="grey", ls=":", lw=1)
plt.text(drift.start, 1.05, " window", color="grey", va="bottom")
plt.xlabel("instance")
plt.ylabel("P(instance from the new concept)")
plt.title("Both transitions start and finish with the window")
plt.legend()
plt.show()


# %% [markdown]
# #### Why the window always contains the transition
#
# * A sigmoid never truly reaches 0 or 1, so left alone it would keep drawing from the old concept long after the drift was supposed to be over.
# * CapyMOA parametrises it from the width instead, so the curve completes inside the window, and clips the last sliver at the edges. In practice that means the transition finishes exactly where you said it would.
# * This is a choice, not a limitation of the shape. If you want a transition that lingers, ask for a **wider window** rather than a curve that overruns a narrow one - the window is the thing the rest of the definition is built around.
# * A custom function is clipped the same way, so it should be written to complete over `0.0` to `1.0` of the window.

# %% [markdown]
# #### Supplying your own transition
#
# * `transition_function` also accepts a callable. It receives **progress through the window** - `0.0` at the start, `1.0` at the end -- and returns the probability that the instance comes from the **new** concept.
# * The result is **clipped to the window**: a function that has not reached 1 by the end is cut off there. If you want a longer transition, widen the window rather than stretching the function.

# %%
# A transition that holds off, then switches quickly near the end.
def late_switch(progress):
    return progress**3


stream_custom = DriftStream(
    stream=[
        Concept(SEA(function=1), num_instances=1000),
        GradualDrift(num_instances=200, transition_function=late_switch),
        Concept(SEA(function=3), num_instances=1000),
    ]
)

print(stream_custom.describe())

# And the effect on where the instances come from, against the default.
stream_default = DriftStream(
    stream=[
        Concept(SEA(function=1), num_instances=1000),
        GradualDrift(num_instances=200),
        Concept(SEA(function=3), num_instances=1000),
    ]
)
print("counts with the default sigmoid:", stream_default.get_concept_counts())
print("counts with progress**3       :", stream_custom.get_concept_counts())

# %% [markdown]
# ## RecurrentConceptDriftStream
#
# Concepts often return: a weekday pattern, a seasonal effect, a fault mode that reappears. `RecurrentConceptDriftStream` builds a stream that cycles through a list of concepts rather than passing through each one once.
#
# The stream it produces is an ordinary `DriftStream`, so everything above applies: the drift metadata, `describe()`, and conversion to MOA. An example of the MOA side is in [Exploring Advanced Features](https://capymoa.org/notebooks/common/advanced_API.html), in the section on creating a synthetic stream with concept drifts from MOA.
#
# **Reference:**
#
# _Gunasekara, N., Pfahringer, B., Gomes, H. M., Bifet, A., & Koh, Y. S. (2024). *Recurrent concept drifts on data streams.* International Joint Conferences on Artificial Intelligence Organization_

# %% [markdown]
# ### Generate a stream with recurrent concepts
#
# In this example, we are simply copying and pasting rather than using `RecurrentConceptDriftStream` to demonstrate that it is possible (but a bit long and might be error prone)

# %%
from capymoa.classifier import OnlineBagging
from capymoa.evaluation import prequential_evaluation
from capymoa.evaluation.visualization import plot_windowed_results
from capymoa.stream.drift import AbruptDrift, DriftStream
from capymoa.stream.generator import RandomTreeGenerator

window_size = 1000
concept_length = 2000
max_instances = concept_length * 6

stream_with_drifts = DriftStream(
    stream=[
        RandomTreeGenerator(tree_random_seed=1),
        AbruptDrift(position=concept_length * 1),
        RandomTreeGenerator(tree_random_seed=2),
        AbruptDrift(position=concept_length * 2),
        RandomTreeGenerator(tree_random_seed=3),
        AbruptDrift(position=concept_length * 3),
        RandomTreeGenerator(tree_random_seed=1, instance_random_seed=2),
        AbruptDrift(position=concept_length * 4),
        RandomTreeGenerator(tree_random_seed=2, instance_random_seed=2),
        AbruptDrift(position=concept_length * 5),
        RandomTreeGenerator(tree_random_seed=3, instance_random_seed=2),
    ]
)

OB = OnlineBagging(schema=stream_with_drifts.get_schema(), ensemble_size=10)

results_stream_with_drifts_OB = prequential_evaluation(
    stream=stream_with_drifts,
    learner=OB,
    window_size=window_size,
    max_instances=max_instances,
)

print(f"Recurrent concept stream CapyMOA:\n{stream_with_drifts}")
plot_windowed_results(results_stream_with_drifts_OB, metric="accuracy")

# %% [markdown]
# ### Use recurrent concept drift API to generate recurrent concepts
#
# The `RecurrentConceptDriftStream` API adds concept meta information for plotting which is not available in the previous example.

# %%
from capymoa.classifier import HoeffdingTree
from capymoa.evaluation import prequential_evaluation
from capymoa.evaluation.visualization import plot_windowed_results
from capymoa.stream.drift import AbruptDrift, GradualDrift, RecurrentConceptDriftStream
from capymoa.stream.generator import LEDGeneratorDrift

# We declare all the concepts
concept1 = LEDGeneratorDrift(number_of_attributes_with_drift=1, instance_random_seed=1)
concept2 = LEDGeneratorDrift(number_of_attributes_with_drift=3, instance_random_seed=1)
concept3 = LEDGeneratorDrift(number_of_attributes_with_drift=5, instance_random_seed=1)
concept4 = LEDGeneratorDrift(number_of_attributes_with_drift=7, instance_random_seed=1)

window_size = 1000
concept_length = 2000
concept_transition_width = 50
max_recurrences_per_concept = 2

concept_list = [concept1, concept2, concept3, concept4]
concept_name_list = ["concept1", "concept2", "concept3", "concept4"]

max_instances = concept_length * len(concept_list) * max_recurrences_per_concept


stream_with_recurrent_concepts = RecurrentConceptDriftStream(
    concept_list=concept_list,
    max_recurrences_per_concept=max_recurrences_per_concept,
    # transition_type_template=AbruptDrift(position=concept_length), # we could use AbruptDrift as well
    transition_type_template=GradualDrift(
        position=concept_length, width=concept_transition_width
    ),
    concept_name_list=concept_name_list,
)

ll = HoeffdingTree(schema=stream_with_recurrent_concepts.get_schema())

results_stream_with_drifts_OB = prequential_evaluation(
    stream=stream_with_recurrent_concepts,
    learner=ll,
    window_size=window_size,
    max_instances=max_instances,
)

plot_windowed_results(results_stream_with_drifts_OB, metric="accuracy")

# %% [markdown]
# ## Working with MOA directly
#
# `DriftStream` composes concepts in Python, so a concept can be any `Stream` - including `NumpyStream`, `CSVStream` and others MOA cannot represent.
#
# Full MOA interoperability is still there for those who want it:
#
# * `to_moa_stream()` converts a `DriftStream` into the equivalent nested MOA `ConceptDriftStream`, when every concept is MOA-backed.
# * A `DriftStream` can also be defined *from* a MOA CLI instead of a list of concepts.
#
# Both are shown in [Exploring Advanced Features](https://capymoa.org/notebooks/common/advanced_API.html), in the section on creating a synthetic stream with concept drifts from MOA, alongside the raw MOA syntax for the same streams.
