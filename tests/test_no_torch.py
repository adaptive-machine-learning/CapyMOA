"""Guard the torch-free import path.

PyTorch is an optional extra (``pip install capymoa[torch]``). The core of
CapyMOA must import and run without it, otherwise ``pip install capymoa`` has to
keep pulling the CUDA stack -- ~2.9 GB on Linux.

These tests simulate PyTorch being absent by blocking it on ``sys.meta_path``,
so they run in the normal CI matrix without needing a separate torch-free
environment. They are the regression guard: the first time someone adds a
top-level ``import torch`` to a core module, this fails.
"""

import subprocess
import sys
import textwrap

import pytest

#: Prelude that makes ``import torch`` fail, as it would without the extra.
_BLOCK_TORCH = """
import sys

class _BlockTorch:
    def find_spec(self, name, path=None, target=None):
        if name == "torch" or name.startswith(("torch.", "torchvision")):
            raise ModuleNotFoundError(f"No module named {name!r}", name=name)
        return None

sys.meta_path.insert(0, _BlockTorch())
"""


def _run_without_torch(body: str) -> subprocess.CompletedProcess:
    """Run ``body`` in a subprocess where importing torch fails."""
    script = _BLOCK_TORCH + textwrap.dedent(body)
    return subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
    )


def test_import_capymoa_without_torch():
    """``import capymoa`` must not need PyTorch."""
    result = _run_without_torch("""
        import capymoa
        import sys
        assert "torch" not in sys.modules, "capymoa imported torch"
        print("OK")
    """)
    assert result.returncode == 0, result.stderr
    assert "OK" in result.stdout


@pytest.mark.parametrize(
    "module",
    [
        "capymoa.base",
        "capymoa.classifier",
        "capymoa.regressor",
        "capymoa.stream",
        "capymoa.datasets",
        "capymoa.evaluation",
        "capymoa.drift",
        "capymoa.anomaly",
        "capymoa.ssl",
    ],
)
def test_core_modules_import_without_torch(module: str):
    """Every core package must import without PyTorch."""
    result = _run_without_torch(f"""
        import sys
        import {module}
        assert "torch" not in sys.modules, "{module} imported torch"
        print("OK")
    """)
    assert result.returncode == 0, result.stderr
    assert "OK" in result.stdout


def test_classification_loop_without_torch():
    """The core stream-learning path must work without PyTorch."""
    result = _run_without_torch("""
        from capymoa.classifier import AdaptiveRandomForestClassifier
        from capymoa.datasets import ElectricityTiny
        from capymoa.evaluation import prequential_evaluation

        stream = ElectricityTiny()
        learner = AdaptiveRandomForestClassifier(schema=stream.get_schema())
        results = prequential_evaluation(stream, learner, max_instances=1000)
        assert results["cumulative"].accuracy() > 50
        print("OK")
    """)
    assert result.returncode == 0, result.stderr
    assert "OK" in result.stdout


def test_torch_backed_name_raises_helpful_error():
    """Reaching a torch-only feature explains how to install it."""
    result = _run_without_torch("""
        from capymoa.exception import OptionalDependencyError
        try:
            from capymoa.base import BatchClassifier  # noqa: F401
        except OptionalDependencyError as err:
            assert "PyTorch is required" in str(err), err
            assert "pip install capymoa[torch]" in str(err), err
            print("OK")
        else:
            raise AssertionError("expected OptionalDependencyError")
    """)
    assert result.returncode == 0, result.stderr
    assert "OK" in result.stdout


def test_torch_backed_names_still_work_when_torch_present():
    """With the extra installed, the public API is unchanged."""
    pytest.importorskip("torch")
    from capymoa.base import Batch, BatchClassifier, BatchRegressor

    assert all(
        isinstance(cls, type) for cls in (Batch, BatchClassifier, BatchRegressor)
    )


@pytest.mark.parametrize(
    "module",
    [
        "capymoa.base",
        "capymoa.classifier",
        "capymoa.stream",
        "capymoa.anomaly",
        "capymoa.ssl",
        "capymoa.drift.detectors",
    ],
)
def test_wildcard_import_without_torch(module: str):
    """``import *`` must not drag in torch-only names.

    ``__all__`` drives ``from package import *``. If the lazy names stayed
    listed, a wildcard import would resolve every torch-backed name and fail
    even for a user who only wanted the core ones.
    """
    result = _run_without_torch(f"""
        from {module} import *  # noqa: F401,F403
        print("OK")
    """)
    assert result.returncode == 0, result.stderr
    assert "OK" in result.stdout


def test_all_drops_lazy_names_without_torch():
    """Without torch, ``__all__`` advertises only what can actually be imported."""
    result = _run_without_torch("""
        import capymoa.base as base
        import capymoa.classifier as classifier

        for name in ("Batch", "BatchClassifier", "BatchRegressor"):
            assert name not in base.__all__, name
        assert "Finetune" not in classifier.__all__

        # Core names are untouched.
        assert "Classifier" in base.__all__
        assert "HoeffdingTree" in classifier.__all__

        # And the names are still reachable by explicit import, with a
        # helpful error rather than silence.
        from capymoa.exception import OptionalDependencyError
        try:
            base.BatchClassifier
        except OptionalDependencyError:
            print("OK")
        else:
            raise AssertionError("expected OptionalDependencyError")
    """)
    assert result.returncode == 0, result.stderr
    assert "OK" in result.stdout


def test_all_keeps_lazy_names_when_torch_present():
    """With torch installed ``__all__`` is unfiltered, so docs stay complete."""
    pytest.importorskip("torch")
    import capymoa.base as base
    import capymoa.classifier as classifier

    for name in ("Batch", "BatchClassifier", "BatchRegressor"):
        assert name in base.__all__, name
    assert "Finetune" in classifier.__all__


def test_abcd_runs_without_torch_on_its_scikit_learn_models():
    """ABCD's ``pca``/``kpca`` models reconstruct with scikit-learn, not PyTorch.

    The autoencoder lives in its own module for exactly this reason, so these
    two configurations have to stay usable in a default install.
    """
    result = _run_without_torch("""
        import numpy as np
        from capymoa.drift.detectors import ABCD

        rng = np.random.default_rng(0)
        data = np.vstack([rng.uniform(0, 0.5, (300, 2)), rng.uniform(0.5, 1.0, (300, 2))])

        for model_id in ("pca", "kpca"):
            detector = ABCD(model_id=model_id)
            for row in data:
                detector.add_element(row)

        # The default must be one of them, or a plain ABCD() breaks the default install.
        ABCD()
        print("OK")
    """)
    assert result.returncode == 0, result.stderr
    assert "OK" in result.stdout


def test_abcd_autoencoder_reports_the_model_not_the_detector():
    """``model_id="ae"`` is the only ABCD configuration needing the extra.

    The error names the model rather than ABCD as a whole, so it does not imply
    the detector is unavailable when only one of its three models is.
    """
    result = _run_without_torch("""
        from capymoa.drift.detectors import ABCD
        from capymoa.exception import OptionalDependencyError

        try:
            ABCD(model_id="ae")
        except OptionalDependencyError as error:
            assert "model_id='ae'" in str(error), str(error)
            print("OK")
        else:
            raise AssertionError("expected OptionalDependencyError")
    """)
    assert result.returncode == 0, result.stderr
    assert "OK" in result.stdout


def test_abcd_stays_in_all_without_torch():
    """ABCD is importable in a torch-free install, so it must stay advertised."""
    result = _run_without_torch("""
        import capymoa.drift.detectors as detectors

        assert "ABCD" in detectors.__all__
        assert detectors.ABCD is not None
        print("OK")
    """)
    assert result.returncode == 0, result.stderr
    assert "OK" in result.stdout


def test_abcd_autoencoder_still_works_when_torch_present():
    """The move to a separate module must not change behaviour with torch installed."""
    pytest.importorskip("torch")
    import numpy as np
    from capymoa.drift.detectors import ABCD

    rng = np.random.default_rng(0)
    data = np.vstack([rng.uniform(0, 0.5, (200, 2)), rng.uniform(0.5, 1.0, (200, 2))])

    detector = ABCD(model_id="ae")
    for row in data:
        detector.add_element(row)
    assert detector.idx == len(data)
