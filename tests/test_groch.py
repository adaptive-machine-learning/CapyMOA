# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for the CapyMOA GroCH classifier wrapper."""

from __future__ import annotations

import pytest

from capymoa.classifier import GroCH, HoeffdingTree
from capymoa.datasets import ElectricityTiny
from capymoa.evaluation import prequential_evaluation


def test_groch_default_electricity_tiny():
    """GroCH runs cleanly on ElectricityTiny with Euclidean distance and no pretraining ARFFs."""
    stream = ElectricityTiny()
    learner = GroCH(
        schema=stream.get_schema(),
        base_learner="trees.HoeffdingTree",
        distance_metric="Euclidean",
        console_log_level=0,
    )
    results = prequential_evaluation(
        stream=stream,
        learner=learner,
        max_instances=1000,
    )
    assert results.cumulative.accuracy() > 50.0
    assert str(learner) == "GroCH"
    assert "baseLearner" in learner.cli_help()


def test_groch_python_kwargs_match_cli():
    """Configuring GroCH via Python kwargs matches the equivalent raw MOA CLI bit-for-bit."""
    stream_a = ElectricityTiny()
    learner_kwargs = GroCH(
        schema=stream_a.get_schema(),
        base_learner=HoeffdingTree(schema=stream_a.get_schema()),
        drift_detection_method="RDDM -o 2.0",
        topology_learner="GNG -l 5 -m 200 -c 2147483647 -b",
        distance_metric="Euclidean",
        dist_threshold=1.0,
        max_group_size=5,
        min_insertion_threshold=100,
        min_w_size_for_drift=20,
        min_topology_size_for_drift=1,
        default_window=1,
        min_window_size=1,
        force_early_drifts=True,
        train_active_on_warn=True,
        train_after_drift_evaluation=True,
        console_log_level=0,
    )
    res_kwargs = prequential_evaluation(
        stream=stream_a,
        learner=learner_kwargs,
        max_instances=1000,
    )

    stream_b = ElectricityTiny()
    cli = (
        "-l (trees.HoeffdingTree) -d (RDDM -o 2.0) "
        "-c (GNG -l 5 -m 200 -c 2147483647 -b) "
        "-s Euclidean -t 1.0 -M 5 -I 100 -w 20 -a 1 -D 1 -z 1 "
        "-F -T -B -V 0"
    )
    learner_cli = GroCH(schema=stream_b.get_schema(), CLI=cli)
    res_cli = prequential_evaluation(
        stream=stream_b,
        learner=learner_cli,
        max_instances=1000,
    )

    assert res_kwargs.cumulative.accuracy() == res_cli.cumulative.accuracy()
    assert res_kwargs.cumulative.kappa() == res_cli.cumulative.kappa()


def test_groch_fast_vs_slow_prequential():
    """Fast Java evaluation loop (optimise=True) matches Python loop (optimise=False)."""
    stream_fast = ElectricityTiny()
    learner_fast = GroCH(
        schema=stream_fast.get_schema(),
        base_learner="trees.HoeffdingTree",
        distance_metric="Euclidean",
        console_log_level=0,
    )
    res_fast = prequential_evaluation(
        stream=stream_fast,
        learner=learner_fast,
        max_instances=500,
        optimise=True,
    )

    stream_slow = ElectricityTiny()
    learner_slow = GroCH(
        schema=stream_slow.get_schema(),
        base_learner="trees.HoeffdingTree",
        distance_metric="Euclidean",
        console_log_level=0,
    )
    res_slow = prequential_evaluation(
        stream=stream_slow,
        learner=learner_slow,
        max_instances=500,
        optimise=False,
    )

    assert res_fast.cumulative.accuracy() == res_slow.cumulative.accuracy()
    assert res_fast.cumulative.kappa() == res_slow.cumulative.kappa()


def test_groch_invalid_options():
    """Invalid categorical options raise ValueError before touching MOA."""
    stream = ElectricityTiny()
    schema = stream.get_schema()

    with pytest.raises(ValueError, match="distance_metric"):
        GroCH(schema=schema, distance_metric="Cosine")  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="number_of_matrices"):
        GroCH(schema=schema, number_of_matrices="2 Matrices")  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="group_replacement_policy"):
        GroCH(schema=schema, group_replacement_policy="LRU")  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="priority_of_recurring_classifiers"):
        GroCH(schema=schema, priority_of_recurring_classifiers="Random")  # type: ignore[arg-type]
