# SPDX-License-Identifier: BSD-3-Clause
from importlib import import_module

_LAZY_REGRESSORS = {
    "SOKNLBT": "._soknl_base_tree",
    "SOKNL": "._soknl",
    "ORTO": "._orto",
    "KNNRegressor": "._knn",
    "FIMTDD": "._fimtdd",
    "ARFFIMTDD": "._arffimtdd",
    "AdaptiveRandomForestRegressor": "._adaptive_random_forest",
    "PassiveAggressiveRegressor": "._passive_aggressive_regressor",
    "SGDRegressor": "._sgd_regressor",
    "ShrubsRegressor": "._shrubs_regressor",
    "FESLRegressor": "._fesl_regressor",
    "OASFRegressor": "._oasf_regressor",
}

_MOA_REGRESSORS = {
    "SOKNLBT",
    "SOKNL",
    "ORTO",
    "KNNRegressor",
    "FIMTDD",
    "ARFFIMTDD",
    "AdaptiveRandomForestRegressor",
}


def __getattr__(name: str):
    if name in _LAZY_REGRESSORS:
        if name in _MOA_REGRESSORS:
            from openmoa._prepare_jpype import _start_jpype

            _start_jpype()
        module = import_module(_LAZY_REGRESSORS[name], package=__name__)
        cls = getattr(module, name)
        globals()[name] = cls
        return cls
    raise AttributeError(f"module 'openmoa.regressor' has no attribute {name!r}")


__all__ = list(_LAZY_REGRESSORS)