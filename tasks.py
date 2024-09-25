"""
tasks.py is a Python file used in the task automation framework Invoke.
It contains a collection of tasks, which are functions that can be executed from the command line.

To execute a task, you can run the `invoke` command followed by the task name.
For example, to build the project, you can run `invoke build`.
"""

from invoke import task
from invoke.collection import Collection
from invoke.context import Context
from pathlib import Path
from typing import List, Optional
from subprocess import run
import wget
from os import cpu_count, environ

IS_CI = (environ.get("CI", "false").lower() == "true")

def all_exist(files: List[str] = None, directories: List[str] = None) -> bool:
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


@task()
def docs_build(ctx: Context, ignore_warnings: bool = False):
    """Build the documentation using Sphinx."""
    warn = "-W" if not ignore_warnings else ""
    nitpicky = "-n" if not ignore_warnings else ""

    doc_dir = Path("docs/_build")
    doc_dir.mkdir(exist_ok=True, parents=True)
    cpu = cpu_count() // 2
    print("Building documentation...")

    ctx.run(f"python -m sphinx build {warn} {nitpicky} --color -E -b html docs {doc_dir}")

    print("-" * 80)
    print("Documentation is built and available at:")
    print(f"  file://{doc_dir.resolve()}/index.html")
    print("You can copy and paste this URL into your browser.")
    print("-" * 80)

@task
def docs_coverage(ctx: Context):
    """Check the coverage of the documentation.
    
    Requires the `interrogate` package.
    """
    ctx.run("python -m interrogate -vv -c pyproject.toml || true")

@task
def docs_clean(ctx: Context):
    """Remove the built documentation."""
    ctx.run("rm -r docs/_build")
    ctx.run("rm docs/api/modules/*")



@task
def download_moa(ctx: Context):
    """Download moa.jar from the web."""
    url = ctx["moa_url"]
    moa_path = Path(ctx["moa_path"])
    if not moa_path.exists():
        moa_path.parent.mkdir(parents=True, exist_ok=True)
        print(f"Downloading moa.jar from : {url}")
        wget.download(url, out=moa_path.resolve().as_posix())
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
    class_path = moa_path.resolve().as_posix()

    if all_exist(
        directories=[
            "src/moa-stubs",
            "src/com-stubs/yahoo/labs/samoa",
            "src/com-stubs/github/javacliparser",
        ]
    ):
        print("Nothing todo: Java stubs already exist.")
        return

    result = run(
        [
            "python", "-m", "stubgenj",
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
    )

    if result.returncode != 0:
        # If we are running in Github action we should be more strict.
        if IS_CI:
            raise Exception("Failed to generate Java stubs.")
        else:
            print("Optional step `invoke build.stubs` failed. Continuing anyway ...")


@task
def clean_stubs(ctx: Context):
    """Remove the Java stubs."""
    ctx.run("rm -r src/moa-stubs src/com-stubs || echo 'Nothing to do: Java stubs do not exist.'")

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
    ctx.run("python -c 'import capymoa; capymoa.about()'")


@task(pre=[clean_stubs, clean_moa])
def clean(ctx: Context):
    """Clean all build artifacts."""


@task(
    help={
        "parallel": "Run the notebooks in parallel.",
        "overwrite": (
            "Overwrite the notebooks with the executed output. Requires ``--slow``."
        ),
        "k_pattern": "Run only the notebooks that match the pattern. Same as `pytest -k`",
        "slow": (
            "Run the notebooks in slow mode by setting the environment variable "
            "`NB_FAST` to `false`."
        ),
        "no_skip": "Do not skip any notebooks.",
    }
)
def test_notebooks(
    ctx: Context,
    parallel: bool = False,
    overwrite: bool = False,
    k_pattern: Optional[str] = None,
    slow: bool = False,
    no_skip: bool = False,
):
    """Run the notebooks and check for errors.

    Uses nbmake https://github.com/treebeardtech/nbmake to execute the notebooks
    and check for errors.
    
    The `--overwrite` flag can be used to overwrite the notebooks with the
    executed output.
    """
    assert not (not slow and overwrite), "You cannot use `--overwrite` with `--fast`."

    # Set the environment variable to run the notebooks in fast mode.
    if not slow:
        environ["NB_FAST"] = "true"
        timeout = 60 * 2
    else:
        timeout = -1

    skip_notebooks = ctx["test_skip_notebooks"]
    if skip_notebooks is None or no_skip:
        skip_notebooks = []
    print(f"Skipping notebooks: {skip_notebooks}")
    cmd = [
        "python -m pytest --nbmake",
        "-x",  # Stop after the first failure
        f"--nbmake-timeout={timeout}",
        "notebooks",
        "--durations=0",  # Show the duration of each notebook
    ]
    cmd += ["-n=auto"] if parallel else []  # Should we run in parallel?
    cmd += (
        ["--overwrite"] if overwrite else []
    )  # Overwrite the notebooks with the executed output
    cmd += ["--deselect " + nb for nb in skip_notebooks]  # Skip some notebooks

    if k_pattern:
        cmd += [f"-k {k_pattern}"]

    ctx.run(" ".join(cmd))


@task
def unittest(ctx: Context, parallel: bool = True):
    """Run the tests using pytest."""
    cmd = [
        "python -m pytest",
        "--doctest-modules",  # Run tests defined in docstrings
        "--durations=0",  # Show the duration of each test
        "-x",  # Stop after the first failure
        "-p no:faulthandler" #jpype can raise irrelevant warnings: https://github.com/jpype-project/jpype/issues/561
    ]
    cmd += ["-n=auto"] if parallel else []
    ctx.run(" ".join(cmd))


@task
def all_tests(ctx: Context, parallel: bool = True):
    """Run all the tests."""
    unittest(ctx, parallel)
    test_notebooks(ctx, parallel)


@task
def commit(ctx: Context):
    """Commit changes using conventional commits.

    Utility wrapper around `python -m commitizen commit`.
    """
    ctx.run("python -m commitizen commit", pty=True)

docs = Collection("docs")
docs.add_task(docs_build, "build")
docs.add_task(docs_clean, "clean")
docs.add_task(docs_coverage, "coverage")

build = Collection("build")
build.add_task(download_moa)
build.add_task(build_stubs, "stubs")
build.add_task(clean_stubs, "clean-stubs")
build.add_task(clean_moa, "clean-moa")
build.add_task(clean)

test = Collection("test")
test.add_task(all_tests, "all", default=True)
test.add_task(test_notebooks, "nb")
test.add_task(unittest, "unit")

ns = Collection()
ns.add_collection(docs)
ns.add_collection(build)
ns.add_collection(test)
ns.add_task(commit)
ns.add_task(refresh_moa)
