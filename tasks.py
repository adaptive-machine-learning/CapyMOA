"""
tasks.py is a Python file used in the task automation framework Invoke.
It contains a collection of tasks, which are functions that can be executed from the command line.

To execute a task, you can run the `invoke` command followed by the task name.
For example, to build the project, you can run `invoke build`.
"""

import os
import shutil
import sys
from os import environ
from pathlib import Path
from subprocess import run

import wget
from invoke import task
from invoke.collection import Collection
from invoke.context import Context
from invoke.exceptions import UnexpectedExit

IS_CI = environ.get("CI", "false").lower() == "true"
COVERAGE_DEFAULT = False
WINDOWS = sys.platform == "win32"

# Scales the pytest/doctest/notebook timeouts below. Release CI runs on
# Windows/macOS runners that are noticeably slower than the Linux runner the
# base timeouts are tuned for, so `release.yml` sets this to `2` to double
# them. Keep the base values here in sync with pyproject.toml's
# `[tool.pytest.ini_options] timeout`, which is the fallback used when pytest
# is invoked directly (bypassing these tasks and thus this factor).
PYTEST_TIMEOUT_FACTOR = float(environ.get("PYTEST_TIMEOUT_FACTOR", "1"))
PYTEST_TIMEOUT = int(90 * PYTEST_TIMEOUT_FACTOR)
NOTEBOOK_FAST_TIMEOUT = int(60 * 3 * PYTEST_TIMEOUT_FACTOR)
NOTEBOOK_SLOW_TIMEOUT = int(60 * 30 * PYTEST_TIMEOUT_FACTOR)
NOTEBOOKS_DIR = Path("notebooks")


def python_exe(profile: str | None = None) -> str:
    if profile:
        return f"python -m cProfile -o {profile}"
    else:
        return "python"


# Set working directory to this file's directory
os.chdir(Path(__file__).parent)


def get_java_home(ctx: Context) -> Path:
    result = ctx.run("java -classpath src/capymoa/jar Home")
    return Path(result.stdout.strip())


def divider(text: str):
    """Print a divider with text centered."""
    print(text.center(88, "-"))


def remove_path(path: Path) -> bool:
    """Delete a file or a directory tree.

    Replaces `rm -r`, which is unavailable on Windows. Returns `True` if
    something was deleted and `False` if the path did not exist.
    """
    if path.is_dir() and not path.is_symlink():
        shutil.rmtree(path)
    elif path.exists() or path.is_symlink():
        path.unlink()
    else:
        return False
    return True


def all_exist(
    files: list[str] | None = None, directories: list[str] | None = None
) -> bool:
    """Check if all files and directories exist."""
    if files:
        for file in files:
            file = Path(file)
            if not file.exists() or not file.is_file():
                return False
    if directories:
        for directory in directories:
            directory = Path(directory)
            if not directory.exists() or not directory.is_dir():
                return False
    return True


def _notebooks_missing_ipynb(notebooks_dir: Path) -> list[Path]:
    """`.py` notebooks that don't have a matching, already-executed `.ipynb`."""
    return [
        py_file
        for py_file in sorted(notebooks_dir.glob("*/*.py"))
        if not py_file.with_suffix(".ipynb").exists()
    ]


@task(help={"ignore_warnings": "Do not treat Sphinx warnings as errors."})
def docs_build(
    ctx: Context,
    ignore_warnings: bool = False,
):
    """Build the documentation using Sphinx."""
    sync_notebooks_to_ipynb(NOTEBOOKS_DIR)

    cmd = []
    cmd += ["python", "-m", "sphinx", "build"]
    cmd += ["--color"]  # color output
    cmd += ["-b", "html"]  # generate html
    if not ignore_warnings:
        cmd += ["-W"]  # warnings as errors
        cmd += ["-n"]  # nitpicky mode

    doc_dir = Path("docs/_build")
    doc_dir.mkdir(exist_ok=True, parents=True)
    cmd += ["docs", doc_dir.as_posix()]  # add source and output directories

    try:
        ctx.run(" ".join(cmd), echo=True)
        print("-" * 80)
        print("Documentation is built and available at:")
        # `as_uri()` produces a valid URL on every platform, including the
        # `file:///C:/...` form Windows browsers expect.
        print(f"  {(doc_dir.resolve() / 'index.html').as_uri()}")
        print("You can copy and paste this URL into your browser.")
    except UnexpectedExit as err:
        print("-" * 80)
        print(
            "Documentation build failed. Here are some tips:\n"
            " - Check the Sphinx output for errors and warnings.\n"
            " - Try running `invoke docs.clean` to remove cached files.\n"
            " - Try running with `--ignore-warnings` to ignore warnings.\n"
            "   The build in CI pipelines will still fail but this might\n"
            "   help you fix the warnings locally.\n"
        )
        # Ensure error code is propagated for CI/CD pipelines
        raise SystemExit(err.result.return_code)


@task(
    help={
        "slow": (
            "Execute notebooks against their full-size datasets, with a "
            "generous timeout (used for release docs). By default notebooks "
            "are executed fast, against mocked/tiny datasets (NB_FAST=true), "
            "which is what PR docs use."
        ),
        "parallel": "Run the notebooks in parallel.",
    }
)
def docs_notebooks(ctx: Context, slow: bool = False, parallel: bool = False):
    """Execute notebooks and bake their outputs into their `.ipynb` files.

    This is a distinct step from `invoke docs.build`, so that a notebook
    execution failure and a Sphinx build failure show up as separate CI
    steps. Uses nbmake's `--overwrite` flag to write real outputs directly
    into the generated `.ipynb` files (rather than a separate cache), so
    they stay easy to open and inspect in Jupyter/VS Code.
    """
    sync_notebooks_to_ipynb(NOTEBOOKS_DIR)

    env = {
        # Consumed by `capymoa._nbmock.is_nb_fast()` in the notebooks themselves.
        "NB_FAST": "false" if slow else "true",
        "CAPYMOA_DATASETS_DIR": os.environ.get("CAPYMOA_DATASETS_DIR", "./data"),
    }
    timeout = NOTEBOOK_SLOW_TIMEOUT if slow else NOTEBOOK_FAST_TIMEOUT

    cmd = [
        "python -m pytest --nbmake --overwrite",
        "-x",  # Stop after the first failure
        f"--nbmake-timeout={timeout}",
        # Disable the global per-test pytest-timeout (see pyproject.toml):
        # it wraps the whole notebook run and would kill legitimately long
        # --slow notebooks; --nbmake-timeout above is the right timeout here.
        "--timeout=0",
        "notebooks",
        "--durations=5",  # Show the duration of each notebook
    ]
    cmd += ["-n=auto"] if parallel else []
    ctx.run(" ".join(cmd), echo=True, env=env)


@task
def docs_clean(ctx: Context):
    """Remove the built documentation."""
    if remove_path(Path("docs/_build")):
        print("Removed docs/_build")
    for path in sorted(Path("docs/api/modules").glob("*")):
        remove_path(path)
        print(f"Removed {path}")


@task
def download_moa(ctx: Context):
    """Download moa.jar from the web."""
    url = ctx["moa_url"]
    moa_path = Path(ctx["moa_path"])
    if not moa_path.exists():
        moa_path.parent.mkdir(parents=True, exist_ok=True)
        print(f"Downloading moa.jar from : {url}")
        wget.download(url, out=str(moa_path.resolve()))
    else:
        print("Nothing todo: `moa.jar` already exists.")


@task(pre=[download_moa])
def build_stubs(ctx: Context):
    """Build Java stubs using stubgenj.

    Uses https://pypi.org/project/stubgenj/ https://gitlab.cern.ch/scripting-tools/stubgenj
    to generate Python stubs for Java classes. This is useful for type checking and
    auto-completion in IDEs. The generated stubs are placed in the `src` directory
    with the `-stubs` suffix.
    """
    moa_path = Path(ctx["moa_path"])
    assert moa_path.exists() and moa_path.is_file()
    class_path = str(moa_path.resolve())

    if all_exist(
        directories=[
            "src/moa-stubs",
            "src/com-stubs/yahoo/labs/samoa",
            "src/com-stubs/github/javacliparser",
        ]
    ):
        print("Nothing todo: Java stubs already exist.")
        return
    cmd = [
        # `sys.executable` is the interpreter running this task, so the
        # stubs are built with the active environment even where `python`
        # is not on `PATH`. No shell is involved, so a path containing
        # spaces (common on Windows) needs no quoting.
        sys.executable,
        "-m",
        "stubgenj",
        f"--classpath={class_path}",
        "--output-dir=src",
        # Options
        "--convert-strings",
        "--no-jpackage-stubs",
        # Names of the packages to generate stubs for
        "moa",
        "com.yahoo.labs.samoa",
        "com.github.javacliparser",
    ]
    # stubgenj has a unreliable java resolution strategy
    environ.setdefault("JAVA_HOME", str(get_java_home(ctx)))
    run(cmd, check=True, env=environ)


@task
def clean_stubs(ctx: Context):
    """Remove the Java stubs."""
    removed = False
    for path in [Path("src/moa-stubs"), Path("src/com-stubs")]:
        if remove_path(path):
            removed = True
            print(f"Removed {path}")
    if not removed:
        print("Nothing to do: Java stubs do not exist.")


@task(pre=[clean_stubs])
def clean_moa(ctx: Context):
    """Remove the moa.jar file."""
    moa_path = Path(ctx["moa_path"])
    if moa_path.exists():
        moa_path.unlink()
        print("Removed moa.jar.")
    else:
        print("Nothing todo: `moa.jar` does not exist.")


@task(pre=[clean_stubs, clean_moa, download_moa, build_stubs])
def refresh_moa(ctx: Context):
    """Replace the moa.jar file with the appropriate version.

    The appropriate version is determined by the `moa_url` variable in the `invoke.yaml` file.
    This is equivalent to the following steps:
    1. Remove the moa.jar file `invoke build.clean-moa`.
    2. Download the moa.jar file `invoke build.download-moa`.
    3. Build the Java stubs. `invoke build.java-stubs`
    """
    # Double quotes, because `cmd.exe` does not treat single quotes as
    # quoting characters and would pass them through to Python.
    ctx.run('python -c "import capymoa; capymoa.about()"')


@task(pre=[clean_stubs, clean_moa])
def clean(ctx: Context):
    """Clean all build artifacts."""


def sync_notebooks_to_ipynb(notebooks_dir: Path) -> None:
    """Bring the `.ipynb` files nbmake and Jupyter need up to date.

    The notebooks are stored as Jupytext `py:percent` scripts (see
    ``notebooks/*/*.py``), which is the source of truth committed to git. The
    generated `.ipynb` siblings are build artifacts (gitignored): nbmake only
    collects `.ipynb` files, and Jupyter needs one to open a kernel.

    Uses `jupytext --sync` where the newer file overwrite the older one.
    """
    for py_file in sorted(notebooks_dir.glob("*/*.py")):
        run(["jupytext", "--sync", str(py_file)], check=True)


@task
def clean_notebooks(ctx: Context):
    """Remove generated notebook artifacts (`.ipynb` files, execution side-effects)."""
    # Note: some `.gif`/`.png` files under `notebooks/*/` are committed
    # documentation assets, not generated artifacts -- don't glob those
    # extensions here.
    patterns = [
        "*/*.ipynb",
        "*/__pycache__",
        "*/*.pdf",
        "*/hs_err_pid*.log",
        "*/runs",
        "*/.ipynb_checkpoints",
        "*/data",
    ]
    for pattern in patterns:
        for path in sorted(NOTEBOOKS_DIR.glob(pattern)):
            if remove_path(path):
                print(f"Removed {path}")


@task(
    help={
        "slow": (
            "Run the notebooks in slow mode by setting the environment variable "
            "`NB_FAST` to `false`."
        ),
        "parallel": "Run the notebooks in parallel.",
    }
)
def notebooks(ctx: Context, slow: bool = False, parallel: bool = False):
    """Deprecated: use `invoke docs.nb` instead.

    Used to run the notebooks through nbmake directly; now delegates to
    `docs.nb`, which does the same thing (bakes real outputs into each
    notebook's `.ipynb` file via `nbmake --overwrite`), so there's a single
    code path for "execute the notebooks" instead of two.
    """
    print(
        "warning: `invoke test.nb` is deprecated, use `invoke docs.nb` "
        "instead. It executes the same notebooks; `test.nb` will be removed "
        "in a future release.",
    )
    docs_notebooks(ctx, slow=slow, parallel=parallel)


@task(
    help={
        "parallel": "Run the tests in parallel.",
        "coverage": "Measure code coverage for the tests.",
        "profile": "If not none, output a cProfile report to the given file.",
    }
)
def pytest(
    ctx: Context,
    parallel: bool = False,
    coverage: bool = COVERAGE_DEFAULT,
    profile: str | None = None,
):
    """Run the tests using pytest.

    Text given after ` -- ` will be passed directly to pytest.
    For example, to run only the tests related to the HoeffdingTree classifier, you can run:

        invoke test.pytest -- tests/test_classifiers.py -k "HoeffdingTree"
    """
    cmd = [
        python_exe(profile),
        "-m pytest",
        "--durations=5",  # Show the duration of each test
        "--exitfirst",  # Exit instantly on first error or failed test
        f"--timeout={PYTEST_TIMEOUT}",
    ]
    cmd += ["--cov"] if coverage else []
    cmd += ["-n=auto"] if parallel else []
    cmd += [ctx.remainder]
    environ.setdefault("COVERAGE_FILE", ".coverage.pytest")
    ctx.run(" ".join(cmd), echo=True, env=environ)


@task(
    help={
        "parallel": "Run the doctests in parallel.",
        "coverage": "Measure code coverage for the doctests.",
        "profile": "If not none, output a cProfile report to the given file.",
    }
)
def doctest(
    ctx: Context,
    parallel: bool = True,
    coverage: bool = COVERAGE_DEFAULT,
    profile: str | None = None,
):
    """Run tests defined in docstrings using pytest.

    Text given after ` -- ` will be passed directly to pytest.
    For example, to run only the tests related to the HoeffdingTree classifier, you can run:

        invoke test.doctest -- src/capymoa/classifier/_hoeffding_tree.py
    """
    cmd = [
        python_exe(profile),
        "-m pytest",
        "--doctest-modules",  # Enable doctest tests
        "--durations=5",  # Show the duration of each test
        "--exitfirst",  # Exit instantly on first error or failed test
        "src/capymoa",  # Don't run tests in the `tests` directory
        f"--timeout={PYTEST_TIMEOUT}",
    ]
    cmd += ["--cov"] if coverage else []
    cmd += ["-n=auto"] if parallel else []
    cmd += [ctx.remainder]  # Add any additional arguments passed to the task
    environ.setdefault("COVERAGE_FILE", ".coverage.doctest")
    ctx.run(" ".join(cmd), echo=True, env=environ)


@task(aliases=["cov-combine"])
def coverage_combine(ctx: Context):
    """Combine coverage data from different sources.

    Discovers every ``.coverage.*`` file in the repo root rather than a fixed
    list, since CI produces a variable number of them: the pytest step's
    ``COVERAGE_FILE`` changes per invocation (see ``pytest`` above) so that
    running it more than once in the same job -- once without PyTorch, once
    with -- doesn't have the second run's coverage data silently overwrite
    the first's.
    """
    covfiles = sorted(p.as_posix() for p in Path(".").glob(".coverage.*"))
    if covfiles:
        ctx.run(" ".join(["python -m coverage combine --keep", *covfiles]), echo=True)


@task(aliases=["cov-report"], pre=[coverage_combine])
def coverage_report(ctx: Context):
    """Generate coverage report."""
    ctx.run("python -m coverage html -i", echo=True)


@task(aliases=["cov-clean"])
def coverage_clean(ctx: Context):
    """Clean coverage data."""
    ctx.run("python -m coverage erase", echo=True)
    if remove_path(Path("htmlcov")):
        print("Removed htmlcov")


@task
def all_tests(ctx: Context, parallel: bool = True, coverage: bool = COVERAGE_DEFAULT):
    """Run all the tests."""
    divider("test.pytest")
    pytest(ctx, parallel, coverage)
    divider("test.doctest")
    doctest(ctx, parallel, coverage)
    divider("test.notebooks")
    notebooks(ctx, slow=False, parallel=parallel)
    if coverage:
        divider("test.cov-report")
        coverage_combine(ctx)
        coverage_report(ctx)


@task
def commit(ctx: Context):
    """Commit changes using conventional commits.

    Utility wrapper around `python -m commitizen commit`.
    """
    print("Running Lint Checks ...")
    ctx.run("python -m ruff check")
    print("Running Format Checks ...")
    ctx.run("python -m ruff format --check")
    # Commitizen's prompt needs a TTY, but Windows has no pty. Invoke falls
    # back automatically, so ask for one only where it exists.
    ctx.run("python -m commitizen commit", pty=not WINDOWS)


@task
def lint(ctx: Context):
    """Lint the code using ruff."""
    ctx.run("python -m ruff check --fix")


@task(aliases=["fmt"])
def format(ctx: Context):
    """Format the code using ruff."""
    ctx.run("python -m ruff format", echo=True)
    ctx.run("python -m ruff check --fix", echo=True)


docs = Collection("docs")
docs.add_task(docs_build, "build", default=True)
docs.add_task(docs_notebooks, "nb")
docs.add_task(docs_clean, "clean")

build = Collection("build")
build.add_task(download_moa)
build.add_task(build_stubs, "stubs")
build.add_task(clean_stubs, "clean-stubs")
build.add_task(clean_moa, "clean-moa")
build.add_task(clean)

test = Collection("test")
test.add_task(all_tests, "all", default=True)
test.add_task(docs_notebooks, "nb")
test.add_task(pytest, "pytest")
test.add_task(doctest, "doctest")
test.add_task(coverage_combine)
test.add_task(coverage_clean)
test.add_task(coverage_report)

# Aggregates clean tasks from the other collections under one namespace,
# alongside `clean.nb` (there's no equivalent elsewhere). `docs.clean` and
# `build.clean` keep working as-is; this is purely additive.
clean_ns = Collection("clean")
clean_ns.add_task(clean_notebooks, "nb")
clean_ns.add_task(docs_clean, "docs")
clean_ns.add_task(clean, "build")

ns = Collection()
ns.add_collection(docs)
ns.add_collection(build)
ns.add_collection(clean_ns)
ns.add_collection(test)
ns.add_task(commit)
ns.add_task(refresh_moa)
ns.add_task(lint)
ns.add_task(format)
