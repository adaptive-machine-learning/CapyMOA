# SPDX-License-Identifier: BSD-3-Clause
"""
Unit tests for OpenMOA UOL (Utilitarian Online Learning) classifiers.

Covers:
- Smoke tests: train + predict does not crash
- predict_proba correctness: valid probability vector
- _ensure_dimension: FOBOS / FTRL don't crash with growing feature indices
- ORF3V: weight update uses correct global feature IDs
- OSLMF: batch accumulation (no training before batch_size instances)
- Reproducibility: same seed => same predictions
"""
import numpy as np
import pytest

from openmoa.stream import NumpyStream
from openmoa.stream import OpenFeatureStream
from openmoa.classifier import (
    FESLClassifier,
    OASFClassifier,
    RSOLClassifier,
    FOBOSClassifier,
    FTRLClassifier,
    OVFMClassifier,
    OSLMFClassifier,
    ORF3VClassifier,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_numpy_stream(n: int, d: int, n_classes: int = 2, seed: int = 0):
    rng = np.random.RandomState(seed)
    X = rng.rand(n, d).astype(np.float32)
    y = rng.randint(0, n_classes, size=n)
    return NumpyStream(X, y, dataset_name="test",
                       feature_names=[f"f{i}" for i in range(d)])


def _assert_valid_proba(proba: np.ndarray, n_classes: int):
    """Check that a probability array is valid."""
    assert proba.shape == (n_classes,), \
        f"Expected shape ({n_classes},), got {proba.shape}"
    assert np.all(proba >= 0.0), "All probabilities must be >= 0"
    assert np.all(proba <= 1.0), "All probabilities must be <= 1"
    np.testing.assert_allclose(proba.sum(), 1.0, atol=1e-5,
                                err_msg="Probabilities must sum to 1")


def _run_prequential(clf, stream, n: int = 100):
    """Run n prequential steps (test-then-train)."""
    for _ in range(n):
        if not stream.has_more_instances():
            break
        inst = stream.next_instance()
        clf.predict(inst)
        clf.train(inst)


# ---------------------------------------------------------------------------
# RSOL
# ---------------------------------------------------------------------------

class TestRSOL:
    def test_smoke(self):
        d, n = 6, 100
        stream = _make_numpy_stream(n, d)
        clf = RSOLClassifier(schema=stream.get_schema(), random_seed=1)
        _run_prequential(clf, stream, n)

    def test_predict_proba(self):
        d = 6
        stream = _make_numpy_stream(50, d)
        clf = RSOLClassifier(schema=stream.get_schema(), random_seed=1)
        for _ in range(30):
            inst = stream.next_instance()
            clf.train(inst)
        inst = stream.next_instance()
        proba = clf.predict_proba(inst)
        _assert_valid_proba(proba, 2)

    def test_binary_only_raises(self):
        mc_stream = _make_numpy_stream(10, 4, n_classes=3)
        with pytest.raises(ValueError):
            RSOLClassifier(schema=mc_stream.get_schema())

    def test_reproducibility(self):
        stream1 = _make_numpy_stream(50, 6, seed=0)
        stream2 = _make_numpy_stream(50, 6, seed=0)
        clf1 = RSOLClassifier(schema=stream1.get_schema(), random_seed=42)
        clf2 = RSOLClassifier(schema=stream2.get_schema(), random_seed=42)
        for _ in range(30):
            clf1.train(stream1.next_instance())
            clf2.train(stream2.next_instance())
        p1 = clf1.predict_proba(stream1.next_instance())
        p2 = clf2.predict_proba(stream2.next_instance())
        np.testing.assert_array_equal(p1, p2)


# ---------------------------------------------------------------------------
# FESL
# ---------------------------------------------------------------------------

class TestFESL:
    def test_smoke(self):
        d, n = 8, 100
        stream = _make_numpy_stream(n, d)
        clf = FESLClassifier(schema=stream.get_schema(), random_seed=1)
        _run_prequential(clf, stream, n)

    def test_predict_proba(self):
        d = 8
        stream = _make_numpy_stream(80, d)
        clf = FESLClassifier(schema=stream.get_schema(), random_seed=1)
        for _ in range(40):
            clf.train(stream.next_instance())
        proba = clf.predict_proba(stream.next_instance())
        _assert_valid_proba(proba, 2)

    def test_with_feature_indices(self):
        """FESL must work when instances carry feature_indices (OpenFeatureStream)."""
        d = 8
        base = _make_numpy_stream(100, d)
        stream = OpenFeatureStream(base, d_min=3, d_max=d,
                                   evolution_pattern="incremental",
                                   total_instances=100)
        clf = FESLClassifier(schema=stream.get_schema(), random_seed=1)
        for _ in range(100):
            if not stream.has_more_instances():
                break
            inst = stream.next_instance()
            clf.predict(inst)
            clf.train(inst)


# ---------------------------------------------------------------------------
# OASF
# ---------------------------------------------------------------------------

class TestOASF:
    def test_smoke(self):
        d, n = 6, 100
        stream = _make_numpy_stream(n, d)
        clf = OASFClassifier(schema=stream.get_schema(), random_seed=1)
        _run_prequential(clf, stream, n)

    def test_predict_proba(self):
        d = 6
        stream = _make_numpy_stream(80, d)
        clf = OASFClassifier(schema=stream.get_schema(), random_seed=1)
        for _ in range(40):
            clf.train(stream.next_instance())
        proba = clf.predict_proba(stream.next_instance())
        _assert_valid_proba(proba, 2)


# ---------------------------------------------------------------------------
# FOBOS
# ---------------------------------------------------------------------------

class TestFOBOS:
    def test_smoke(self):
        d, n = 6, 100
        stream = _make_numpy_stream(n, d)
        clf = FOBOSClassifier(schema=stream.get_schema(), random_seed=1)
        _run_prequential(clf, stream, n)

    def test_predict_proba_binary(self):
        d = 6
        stream = _make_numpy_stream(80, d)
        clf = FOBOSClassifier(schema=stream.get_schema(), random_seed=1)
        for _ in range(40):
            clf.train(stream.next_instance())
        proba = clf.predict_proba(stream.next_instance())
        _assert_valid_proba(proba, 2)

    def test_growing_feature_indices_no_crash(self):
        """A1: _ensure_dimension must handle growing feature indices from OpenFeatureStream."""
        d = 8
        base = _make_numpy_stream(100, d)
        stream = OpenFeatureStream(base, d_min=2, d_max=d,
                                   evolution_pattern="incremental",
                                   total_instances=100)
        clf = FOBOSClassifier(schema=stream.get_schema(), random_seed=1)
        # This must not raise IndexError even as dim grows
        for _ in range(100):
            if not stream.has_more_instances():
                break
            inst = stream.next_instance()
            clf.predict(inst)
            clf.train(inst)


# ---------------------------------------------------------------------------
# FTRL
# ---------------------------------------------------------------------------

class TestFTRL:
    def test_smoke_binary(self):
        d, n = 6, 100
        stream = _make_numpy_stream(n, d)
        clf = FTRLClassifier(schema=stream.get_schema(), random_seed=1)
        _run_prequential(clf, stream, n)

    def test_smoke_multiclass(self):
        d, n = 6, 100
        stream = _make_numpy_stream(n, d, n_classes=3)
        clf = FTRLClassifier(schema=stream.get_schema(), random_seed=1)
        for _ in range(n):
            if not stream.has_more_instances():
                break
            inst = stream.next_instance()
            clf.predict(inst)
            clf.train(inst)

    def test_predict_proba_binary(self):
        d = 6
        stream = _make_numpy_stream(80, d)
        clf = FTRLClassifier(schema=stream.get_schema(), random_seed=1)
        for _ in range(40):
            clf.train(stream.next_instance())
        proba = clf.predict_proba(stream.next_instance())
        _assert_valid_proba(proba, 2)

    def test_predict_proba_multiclass(self):
        d, n_classes = 6, 3
        stream = _make_numpy_stream(80, d, n_classes=n_classes)
        clf = FTRLClassifier(schema=stream.get_schema(), random_seed=1)
        for _ in range(40):
            clf.train(stream.next_instance())
        proba = clf.predict_proba(stream.next_instance())
        _assert_valid_proba(proba, n_classes)

    def test_growing_feature_indices_no_crash(self):
        """A1: _ensure_dimension must handle growing feature indices from OpenFeatureStream."""
        d = 8
        base = _make_numpy_stream(100, d)
        stream = OpenFeatureStream(base, d_min=2, d_max=d,
                                   evolution_pattern="incremental",
                                   total_instances=100)
        clf = FTRLClassifier(schema=stream.get_schema(), random_seed=1)
        for _ in range(100):
            if not stream.has_more_instances():
                break
            inst = stream.next_instance()
            clf.predict(inst)
            clf.train(inst)

    def test_regression_raises(self):
        rng = np.random.RandomState(0)
        X = rng.rand(10, 2).astype(np.float32)
        y = rng.rand(10).astype(np.float32)
        reg_stream = NumpyStream(X, y, dataset_name="reg", target_type="numeric")
        with pytest.raises(ValueError):
            FTRLClassifier(schema=reg_stream.get_schema())


# ---------------------------------------------------------------------------
# ORF3V
# ---------------------------------------------------------------------------

class TestORF3V:
    def test_smoke(self):
        d, n = 6, 100
        stream = _make_numpy_stream(n, d, n_classes=3)
        clf = ORF3VClassifier(schema=stream.get_schema(), random_seed=1)
        _run_prequential(clf, stream, n)

    def test_predict_proba(self):
        d, n_classes = 6, 3
        stream = _make_numpy_stream(80, d, n_classes=n_classes)
        clf = ORF3VClassifier(schema=stream.get_schema(), random_seed=1)
        for _ in range(40):
            clf.train(stream.next_instance())
        proba = clf.predict_proba(stream.next_instance())
        _assert_valid_proba(proba, n_classes)

    def test_global_feature_ids_used_in_weight_update(self):
        """A2: weight updates must key on global feature_indices, not local position."""
        d = 8
        base = _make_numpy_stream(60, d, n_classes=3)
        stream = OpenFeatureStream(base, d_min=3, d_max=d,
                                   evolution_pattern="incremental",
                                   total_instances=60)
        clf = ORF3VClassifier(schema=stream.get_schema(), random_seed=1)

        # The weight keys must be global IDs (integers 0..d-1), not local positions
        for _ in range(30):
            if not stream.has_more_instances():
                break
            inst = stream.next_instance()
            clf.train(inst)

        # After training, weight keys should be <= d (global IDs), not > d
        for feature_id in clf.weights:
            assert feature_id < d, \
                f"weight key {feature_id} exceeds d={d}: using local position instead of global ID"


# ---------------------------------------------------------------------------
# OVFM
# ---------------------------------------------------------------------------

class TestOVFM:
    def test_smoke(self):
        d, n = 6, 100
        stream = _make_numpy_stream(n, d)
        clf = OVFMClassifier(schema=stream.get_schema(), random_seed=1)
        _run_prequential(clf, stream, n)

    def test_predict_proba(self):
        d = 6
        stream = _make_numpy_stream(80, d)
        clf = OVFMClassifier(schema=stream.get_schema(), random_seed=1)
        for _ in range(50):
            clf.train(stream.next_instance())
        proba = clf.predict_proba(stream.next_instance())
        _assert_valid_proba(proba, 2)


# ---------------------------------------------------------------------------
# OSLMF
# ---------------------------------------------------------------------------

class TestOSLMF:
    def test_smoke(self):
        d, n = 6, 120
        stream = _make_numpy_stream(n, d)
        clf = OSLMFClassifier(schema=stream.get_schema(), batch_size=50,
                              random_seed=1)
        # Must not crash even before first full batch is processed
        for _ in range(n):
            if not stream.has_more_instances():
                break
            inst = stream.next_instance()
            clf.predict(inst)
            clf.train(inst)

    def test_not_trained_before_first_batch(self):
        """A5: model should not be trained until the first full batch accumulates."""
        d = 6
        stream = _make_numpy_stream(200, d)
        batch_size = 50
        clf = OSLMFClassifier(schema=stream.get_schema(), batch_size=batch_size,
                              random_seed=1)

        # Feed batch_size - 1 instances; model should still be untrained
        for _ in range(batch_size - 1):
            clf.train(stream.next_instance())

        assert not clf._trained or clf._num_updates == 0, \
            "No weight updates should occur before the first full batch"

    def test_predict_proba_after_training(self):
        d = 6
        stream = _make_numpy_stream(200, d)
        clf = OSLMFClassifier(schema=stream.get_schema(), batch_size=50,
                              random_seed=1)
        # Train two full batches
        for _ in range(100):
            clf.train(stream.next_instance())
        proba = clf.predict_proba(stream.next_instance())
        _assert_valid_proba(proba, 2)

    def test_binary_only_raises(self):
        mc_stream = _make_numpy_stream(10, 4, n_classes=3)
        with pytest.raises(ValueError):
            OSLMFClassifier(schema=mc_stream.get_schema())
