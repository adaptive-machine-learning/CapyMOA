"""The nbmock module provides support for mocking datasets to speed up testing."""

from os import environ


def mock_datasets():
    """Mock the datasets to use the tiny versions for testing."""
    from unittest import mock

    from capymoa.datasets import CovtypeTiny, ElectricityTiny, FriedTiny
    from capymoa.ocl.datasets import TinySplitMNIST

    mock.patch("capymoa.datasets.Electricity", ElectricityTiny).start()
    mock.patch("capymoa.datasets.Covtype", CovtypeTiny).start()
    mock.patch("capymoa.datasets.Fried", FriedTiny).start()
    mock.patch("capymoa.ocl.datasets.SplitMNIST", TinySplitMNIST).start()


def override_prequential_evaluation(max_instances: int = 100):
    """Monkeypatch the evaluation functions to limit the number of instances.

    This is useful for testing purposes to speed up the evaluation. Import the
    functions after calling this.
    """
    import importlib

    targets = [
        ("capymoa.evaluation", "prequential_evaluation"),
        ("capymoa.classifier", "evaluate_classifier"),
        ("capymoa.classifier", "evaluate_classifiers"),
        ("capymoa.regressor", "evaluate_regressor"),
        ("capymoa.regressor", "evaluate_regressors"),
        ("capymoa.uncertainty", "evaluate_prediction_interval"),
        ("capymoa.uncertainty", "evaluate_prediction_intervals"),
        ("capymoa.anomaly", "evaluate_anomaly"),
        ("capymoa.anomaly", "evaluate_anomaly_detectors"),
        ("capymoa.ssl", "evaluate_ssl"),
    ]

    def limited(function):
        def wrapper(*args, **kwargs):
            kwargs["max_instances"] = max_instances
            return function(*args, **kwargs)

        return wrapper

    for module_name, name in targets:
        module = importlib.import_module(module_name)
        setattr(module, name, limited(getattr(module, name)))


def is_nb_fast() -> bool:
    """Should the notebook be run with faster settings.

    Some notebooks are slow to run because they use large datasets and run
    for many iterations. This is good for documentation purposes but not for
    testing. This function returns True if the notebook should be run with
    faster settings.

    Care should be taken to hide cells in capymoa.org that are meant for testing
    only. This is done by tagging the cell ``remove-cell``
    (e.g. ``# %% tags=["remove-cell"]``).
    See: https://myst-nb.readthedocs.io/en/latest/render/hiding.html
    """
    return environ.get("NB_FAST", "") not in ("", "0", "false", "False")
