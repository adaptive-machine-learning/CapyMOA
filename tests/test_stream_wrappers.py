# SPDX-License-Identifier: BSD-3-Clause
"""
Unit tests for OpenMOA stream wrappers.

Covers:
- feature_indices attachment (OpenFeatureStream)
- NaN placement / fixed vector (TrapezoidalStream)
- stochastic feature masking (CapriciousStream)
- phased feature evolution (EvolvableStream)
- shuffle completeness + restart reproducibility (ShuffledStream)
- restart() RNG reset (all wrappers)
- _calc_eds_boundaries formula
"""
import numpy as np
import pytest

from openmoa.stream import NumpyStream, Schema
from openmoa.stream import (
    OpenFeatureStream,
    TrapezoidalStream,
    CapriciousStream,
    EvolvableStream,
    ShuffledStream,
)
from openmoa.stream.stream_wrapper import _calc_eds_boundaries


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_numpy_stream(n=200, d=8, n_classes=2, seed=0):
    """Build a small NumpyStream for wrapper tests."""
    rng = np.random.RandomState(seed)
    X = rng.rand(n, d).astype(np.float32)
    y = rng.randint(0, n_classes, size=n)
    return NumpyStream(X, y, dataset_name="test",
                       feature_names=[f"f{i}" for i in range(d)])


# ---------------------------------------------------------------------------
# _calc_eds_boundaries
# ---------------------------------------------------------------------------

class TestCalcEdsBoundaries:
    def test_two_segments_no_overlap(self):
        bounds = _calc_eds_boundaries(100, 2, 0.0)
        # 3 stages: stable0, overlap, stable1
        assert len(bounds) == 3
        assert bounds[-1] == 100

    def test_two_segments_equal_overlap(self):
        # total = L*(2 + 1*1) = 3L  =>  L=10 for total=30
        bounds = _calc_eds_boundaries(30, 2, 1.0)
        assert len(bounds) == 3
        assert bounds[-1] == 30
        # stable0 ends at 10, overlap ends at 20, stable1 ends at 30
        assert bounds[0] == 10
        assert bounds[1] == 20

    def test_three_segments(self):
        bounds = _calc_eds_boundaries(100, 3, 0.0)
        assert len(bounds) == 5       # 2*3 - 1 = 5
        assert bounds[-1] == 100

    def test_last_boundary_always_equals_total(self):
        for n in [50, 100, 137, 500]:
            bounds = _calc_eds_boundaries(n, 2, 0.5)
            assert bounds[-1] == n


# ---------------------------------------------------------------------------
# OpenFeatureStream
# ---------------------------------------------------------------------------

class TestOpenFeatureStream:
    def test_feature_indices_attached(self):
        base = _make_numpy_stream(n=50, d=8)
        stream = OpenFeatureStream(base, d_min=3, d_max=8,
                                   evolution_pattern="pyramid",
                                   total_instances=50)
        inst = stream.next_instance()
        assert hasattr(inst, "feature_indices"), "feature_indices must be attached"
        assert len(inst.feature_indices) == len(inst.x)

    def test_incremental_dimension_grows(self):
        base = _make_numpy_stream(n=100, d=8)
        stream = OpenFeatureStream(base, d_min=2, d_max=8,
                                   evolution_pattern="incremental",
                                   total_instances=100)
        first = stream.next_instance()
        for _ in range(98):
            stream.next_instance()
        last = stream.next_instance()
        assert len(first.x) <= len(last.x), \
            "incremental: last instance should have >= features than first"

    def test_decremental_dimension_shrinks(self):
        base = _make_numpy_stream(n=100, d=8)
        stream = OpenFeatureStream(base, d_min=2, d_max=8,
                                   evolution_pattern="decremental",
                                   total_instances=100)
        first = stream.next_instance()
        for _ in range(98):
            stream.next_instance()
        last = stream.next_instance()
        assert len(first.x) >= len(last.x), \
            "decremental: last instance should have <= features than first"

    def test_eds_feature_indices_within_range(self):
        base = _make_numpy_stream(n=100, d=8)
        stream = OpenFeatureStream(base, d_min=1, d_max=8,
                                   evolution_pattern="eds",
                                   n_segments=2, overlap_ratio=1.0,
                                   total_instances=100)
        for _ in range(100):
            inst = stream.next_instance()
            if inst is None:
                break
            assert np.all(inst.feature_indices < 8), \
                "All feature IDs must be within [0, d_max)"
            assert len(inst.feature_indices) == len(inst.x)

    def test_restart_reproducibility(self):
        base = _make_numpy_stream(n=50, d=8)
        stream = OpenFeatureStream(base, d_min=2, d_max=8,
                                   evolution_pattern="pyramid",
                                   total_instances=50, random_seed=7)
        run1 = [tuple(stream.next_instance().feature_indices) for _ in range(10)]
        stream.restart()
        run2 = [tuple(stream.next_instance().feature_indices) for _ in range(10)]
        assert run1 == run2, "restart() must produce identical feature_indices sequence"

    def test_invalid_dmin_gt_dmax(self):
        base = _make_numpy_stream(n=20, d=8)
        with pytest.raises(ValueError):
            OpenFeatureStream(base, d_min=6, d_max=4, total_instances=20)

    def test_invalid_dmax_exceeds_original(self):
        base = _make_numpy_stream(n=20, d=8)
        with pytest.raises(ValueError):
            OpenFeatureStream(base, d_max=20, total_instances=20)


# ---------------------------------------------------------------------------
# TrapezoidalStream
# ---------------------------------------------------------------------------

class TestTrapezoidalStream:
    def test_vector_length_fixed(self):
        """Output vector must always have length d_max (with NaN for inactive)."""
        d = 8
        base = _make_numpy_stream(n=100, d=d)
        stream = TrapezoidalStream(base, d_min=2, d_max=d,
                                   evolution_mode="ordered",
                                   total_instances=100)
        for _ in range(100):
            inst = stream.next_instance()
            if inst is None:
                break
            assert len(inst.x) == d, "vector must always have length d_max"

    def test_inactive_features_are_nan(self):
        """Early instances should have some NaN (not all features active yet)."""
        d = 8
        base = _make_numpy_stream(n=100, d=d)
        stream = TrapezoidalStream(base, d_min=1, d_max=d,
                                   evolution_mode="ordered",
                                   total_instances=100)
        inst = stream.next_instance()
        x = np.array(inst.x, dtype=float)
        assert np.any(np.isnan(x)), "early instance should have inactive (NaN) features"

    def test_restart_reproducibility(self):
        d = 8
        base = _make_numpy_stream(n=50, d=d)
        stream = TrapezoidalStream(base, d_min=2, d_max=d,
                                   evolution_mode="random",
                                   total_instances=50, random_seed=42)
        run1 = [np.isnan(np.array(stream.next_instance().x, dtype=float)).tolist()
                for _ in range(10)]
        stream.restart()
        run2 = [np.isnan(np.array(stream.next_instance().x, dtype=float)).tolist()
                for _ in range(10)]
        assert run1 == run2, "restart() must reproduce identical NaN patterns"


# ---------------------------------------------------------------------------
# CapriciousStream
# ---------------------------------------------------------------------------

class TestCapriciousStream:
    def test_vector_length_fixed(self):
        d = 8
        base = _make_numpy_stream(n=100, d=d)
        stream = CapriciousStream(base, missing_ratio=0.3,
                                  total_instances=100, random_seed=1)
        for _ in range(100):
            inst = stream.next_instance()
            if inst is None:
                break
            assert len(inst.x) == d

    def test_missing_ratio_zero_no_nan(self):
        d = 6
        base = _make_numpy_stream(n=50, d=d)
        stream = CapriciousStream(base, missing_ratio=0.0,
                                  total_instances=50, random_seed=1)
        for _ in range(50):
            inst = stream.next_instance()
            if inst is None:
                break
            x = np.array(inst.x, dtype=float)
            assert not np.any(np.isnan(x)), "missing_ratio=0 should produce no NaN"

    def test_missing_ratio_high_produces_nans(self):
        """With missing_ratio close to 1, most instances should have NaN."""
        d = 10
        base = _make_numpy_stream(n=200, d=d)
        stream = CapriciousStream(base, missing_ratio=0.9,
                                  total_instances=200, random_seed=0)
        nan_counts = []
        for _ in range(200):
            inst = stream.next_instance()
            if inst is None:
                break
            x = np.array(inst.x, dtype=float)
            nan_counts.append(np.sum(np.isnan(x)))
        assert np.mean(nan_counts) > 3, "high missing_ratio should produce many NaN on average"

    def test_restart_reproducibility(self):
        d = 8
        base = _make_numpy_stream(n=50, d=d)
        stream = CapriciousStream(base, missing_ratio=0.4,
                                  total_instances=50, random_seed=5)
        run1 = [np.isnan(np.array(stream.next_instance().x, dtype=float)).tolist()
                for _ in range(20)]
        stream.restart()
        run2 = [np.isnan(np.array(stream.next_instance().x, dtype=float)).tolist()
                for _ in range(20)]
        assert run1 == run2


# ---------------------------------------------------------------------------
# EvolvableStream
# ---------------------------------------------------------------------------

class TestEvolvableStream:
    def test_basic_runs_without_error(self):
        d = 8
        base = _make_numpy_stream(n=100, d=d)
        stream = EvolvableStream(base, total_instances=100, random_seed=0)
        count = 0
        while stream.has_more_instances():
            inst = stream.next_instance()
            if inst is None:
                break
            count += 1
        assert count == 100

    def test_restart_reproducibility(self):
        d = 8
        base = _make_numpy_stream(n=60, d=d)
        stream = EvolvableStream(base, total_instances=60, random_seed=3)
        run1 = [np.isnan(np.array(stream.next_instance().x, dtype=float)).tolist()
                for _ in range(20)]
        stream.restart()
        run2 = [np.isnan(np.array(stream.next_instance().x, dtype=float)).tolist()
                for _ in range(20)]
        assert run1 == run2


# ---------------------------------------------------------------------------
# ShuffledStream
# ---------------------------------------------------------------------------

class TestShuffledStream:
    def test_all_instances_produced_once(self):
        n = 80
        base = _make_numpy_stream(n=n, d=6)
        stream = ShuffledStream(base, random_seed=42)
        count = 0
        while stream.has_more_instances():
            inst = stream.next_instance()
            if inst is None:
                break
            count += 1
        assert count == n, "ShuffledStream must produce all instances exactly once"

    def test_restart_same_order(self):
        n = 40
        base = _make_numpy_stream(n=n, d=6, seed=1)
        stream = ShuffledStream(base, random_seed=99)
        run1 = [list(stream.next_instance().x) for _ in range(10)]
        stream.restart()
        run2 = [list(stream.next_instance().x) for _ in range(10)]
        assert run1 == run2, "ShuffledStream restart() must reproduce the same shuffle"

    def test_different_seed_different_order(self):
        n = 40
        base1 = _make_numpy_stream(n=n, d=6, seed=0)
        base2 = _make_numpy_stream(n=n, d=6, seed=0)
        s1 = ShuffledStream(base1, random_seed=1)
        s2 = ShuffledStream(base2, random_seed=2)
        order1 = [list(s1.next_instance().x) for _ in range(n)]
        order2 = [list(s2.next_instance().x) for _ in range(n)]
        assert order1 != order2, "Different seeds must produce different shuffles"
