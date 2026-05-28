# SPDX-License-Identifier: BSD-3-Clause
"""
OpenMOA — minimal demo for paper.

Shows how to run a UOL classifier on an evolving feature stream
in fewer than 20 lines of application code.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from openmoa.datasets import Electricity             # benchmark dataset
from openmoa.stream import EvolvingFeatureStream     # feature-evolution wrapper
from openmoa.classifier import ORF3VClassifier       # UOL classifier

# ── 1. Build an evolving feature stream (TDS pattern) ────────────────────────
stream = EvolvingFeatureStream(
    base_stream=Electricity(),    # base dataset (45 312 instances, 8 features)
    evolution_pattern="tds",      # features appear gradually over time
    d_max=6,                      # maximum active feature dimensions
    total_instances=10000,        # number of instances to generate
)

# ── 2. Initialise the classifier ─────────────────────────────────────────────
learner = ORF3VClassifier(
    schema=stream.get_schema(),   # stream schema (feature/class info)
    n_stumps=20,                  # number of decision stumps per tree
    alpha=0.3,                    # learning rate for weight updates
    grace_period=100,             # instances before splitting a node
)

# ── 3. Prequential evaluation (test-then-train) ───────────────────────────────
correct, total = 0, 0
while stream.has_more_instances():
    instance = stream.next_instance()
    pred = learner.predict(instance)   # predict before seeing the label
    learner.train(instance)            # update model with true label
    correct += int(pred == instance.y_index)
    total += 1

print(f"Prequential accuracy: {correct / total:.4f}  ({total} instances)")
