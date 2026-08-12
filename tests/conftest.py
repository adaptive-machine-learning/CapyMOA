"""This conftest.py contains pytest configuration and fixtures shared across all tests.

- https://docs.pytest.org/en/stable/reference/fixtures.html#conftest-py-sharing-fixtures-across-multiple-files
"""

import inspect
import os

import pytest
from _pytest.mark.expression import Expression
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

    def markskip(mark: str, *, reason: str | None = None) -> None:
        """Skip the current test/module if `-m` doesn't select `mark`.

        Called at module scope, this also applies `pytestmark = pytest.mark.<mark>`
        to the caller's module -- equivalent to writing that line yourself --
        so a single call both marks the module and guards its imports.
        """
        __tracebackhide__ = True
        caller = inspect.stack()[1].frame
        if caller.f_code.co_name == "<module>":
            existing = caller.f_globals.get("pytestmark", [])
            if not isinstance(existing, list):
                existing = [existing]
            caller.f_globals["pytestmark"] = [*existing, getattr(pytest.mark, mark)]

        markexpr = config.getoption("markexpr")
        selected = not markexpr or Expression.compile(markexpr).evaluate(
            lambda name: name == mark
        )
        if not selected:
            raise Skipped(
                reason or f"tests marked {mark!r} are excluded by -m",
                allow_module_level=True,
            )

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
