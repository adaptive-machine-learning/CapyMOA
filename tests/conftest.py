"""This conftest.py contains pytest configuration and fixtures shared across all tests.

- https://docs.pytest.org/en/stable/reference/fixtures.html#conftest-py-sharing-fixtures-across-multiple-files
"""

import os

import pytest
from _pytest.mark.expression import Expression
from _pytest.mark.structures import MarkDecorator
from _pytest.outcomes import Skipped

from capymoa.datasets._source_list import SOURCE_LIST
from capymoa.datasets._utils import (
    get_download_dir,
    download_unpacked,
    is_already_downloaded,
)


def pytest_configure(config):
    # Working directory used to be wherever pytest was invoked from, breaking
    # relative paths in tests.
    os.chdir(config.rootpath)

    def markskip(mark: str, *, reason: str | None = None) -> MarkDecorator:
        """Return `pytest.mark.<mark>`, or skip if `-m` doesn't select it.

        Assign the result to `pytestmark` to mark and, when excluded, skip a
        whole module in one line: `pytestmark = pytest.markskip("torch")`.
        Call it bare (return value discarded) inside a lazy constructor to
        guard just one case of an otherwise mixed file instead.
        """
        __tracebackhide__ = True
        markexpr = config.getoption("markexpr")
        selected = not markexpr or Expression.compile(markexpr).evaluate(
            lambda name: name == mark
        )
        if not selected:
            raise Skipped(
                reason or f"tests marked {mark!r} are excluded by -m",
                allow_module_level=True,
            )
        return getattr(pytest.mark, mark)

    # --import-mode=importlib makes `from conftest import markskip`
    # unreliable from other test files, so expose it the way
    # pytest.importorskip is exposed.
    pytest.markskip = markskip


def download_required_testfiles():
    csvs = ["ElectricityTiny", "FriedTiny"]
    arffs = ["ElectricityTiny", "FriedTiny"]
    download_dir = get_download_dir().absolute()

    for dataset in csvs:
        url = SOURCE_LIST[dataset].csv
        if not is_already_downloaded(url, download_dir):
            download_unpacked(url, download_dir)

    for dataset in arffs:
        url = SOURCE_LIST[dataset].arff
        if not is_already_downloaded(url, download_dir):
            download_unpacked(url, download_dir)


download_required_testfiles()
