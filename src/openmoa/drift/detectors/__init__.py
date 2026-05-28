# SPDX-License-Identifier: BSD-3-Clause
from importlib import import_module

_LAZY_DETECTORS = {
    "ADWIN": ".adwin",
    "CUSUM": ".cusum",
    "DDM": ".ddm",
    "EWMAChart": ".ewma_chart",
    "GeometricMovingAverage": ".geometric_ma",
    "HDDMAverage": ".hddm_a",
    "HDDMWeighted": ".hddm_w",
    "PageHinkley": ".page_hinkley",
    "RDDM": ".rddm",
    "SEED": ".seed",
    "STEPD": ".stepd",
    "ABCD": ".abcd",
}

_MOA_DETECTORS = set(_LAZY_DETECTORS) - {"ABCD"}


def __getattr__(name: str):
    if name in _LAZY_DETECTORS:
        if name in _MOA_DETECTORS:
            from openmoa._prepare_jpype import _start_jpype

            _start_jpype()
        try:
            module = import_module(_LAZY_DETECTORS[name], package=__name__)
        except ModuleNotFoundError as exc:
            if exc.name == "torch" and name == "ABCD":
                raise ImportError(
                    "ABCD requires PyTorch. Install PyTorch before importing "
                    "openmoa.drift.detectors.ABCD."
                ) from exc
            raise
        cls = getattr(module, name)
        globals()[name] = cls
        return cls
    raise AttributeError(f"module 'openmoa.drift.detectors' has no attribute {name!r}")


__all__ = list(_LAZY_DETECTORS)