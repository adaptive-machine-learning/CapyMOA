"""This conftest.py contains pytest configuration and fixtures shared across all tests.

- https://docs.pytest.org/en/stable/reference/fixtures.html#conftest-py-sharing-fixtures-across-multiple-files
"""

import os


def pytest_configure(config):
    os.chdir(config.rootpath)
    """Ensure that the working directory is the root of the project.

    We added this because previously, the working directory was wherever the
    pytest command was run from. This caused issues with relative paths in the
    tests.
    """
