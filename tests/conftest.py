"""This conftest.py contains pytest configuration and fixtures shared across all tests.

- https://docs.pytest.org/en/stable/reference/fixtures.html#conftest-py-sharing-fixtures-across-multiple-files
"""

import os
from capymoa.datasets._source_list import SOURCE_LIST
from capymoa.datasets._utils import (
    get_download_dir,
    download_extract,
    is_already_downloaded,
)


def pytest_configure(config):
    os.chdir(config.rootpath)
    """Ensure that the working directory is the root of the project.

    We added this because previously, the working directory was wherever the
    pytest command was run from. This caused issues with relative paths in the
    tests.
    """


def download_required_testfiles():
    csvs = ["ElectricityTiny", "FriedTiny"]
    arffs = ["ElectricityTiny", "FriedTiny"]
    download_dir = get_download_dir().absolute()

    for dataset in csvs:
        url = SOURCE_LIST[dataset].csv
        if not is_already_downloaded(url, download_dir):
            download_extract(url, download_dir)

    for dataset in arffs:
        url = SOURCE_LIST[dataset].arff
        if not is_already_downloaded(url, download_dir):
            download_extract(url, download_dir)


download_required_testfiles()
