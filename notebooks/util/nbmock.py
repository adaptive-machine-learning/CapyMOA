# SPDX-License-Identifier: BSD-3-Clause
"""The nbmock module provides support for mocking datasets to speed up testing."""

from os import environ


def mock_datasets():
    """Mock the datasets to use the tiny versions for testing."""
    import unittest.mock as mock
    from openmoa.datasets import ElectricityTiny, CovtypeTiny, FriedTiny
    from openmoa.ocl.datasets import TinySplitMNIST

    mock.patch("openmoa.datasets.Electricity", ElectricityTiny).start()
    mock.patch("openmoa.datasets.Covtype", CovtypeTiny).start()
    mock.patch("openmoa.datasets.Fried", FriedTiny).start()
    mock.patch("openmoa.ocl.datasets.SplitMNIST", TinySplitMNIST).start()


def is_nb_fast() -> bool:
    """Should the notebook be run with faster settings.

    Some notebooks are slow to run because they use large datasets and run
    for many iterations. This is good for documentation purposes but not for
    testing. This function returns True if the notebook should be run with
    faster settings.

    Care should be taken to hide cells in openmoa.org that are meant for testing
    only. This is done by adding ``"nbsphinx": "hidden"`` to the cell metadata.
    See: https://nbsphinx.readthedocs.io/en/0.9.3/hidden-cells.html
    """
    return bool(environ.get("NB_FAST", False))
