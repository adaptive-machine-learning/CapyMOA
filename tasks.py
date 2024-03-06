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
from typing import List
import wget


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
def docs_build(ctx: Context):
    """Build the documentation using Sphinx."""
    doc_dir = Path("docs/_build/html")
    doc_dir.mkdir(exist_ok=True, parents=True)
    print("Building documentation...")
    ctx.run(f"python -m sphinx build -b html docs {doc_dir}")

    print("-" * 80)
    print("Documentation is built and available at:")
    print(f"  file://{doc_dir.resolve()}/index.html")
    print("You can copy and paste this URL into your browser.")
    print("-" * 80)


@task
def docs_clean(ctx: Context):
    """Remove the built documentation."""
    ctx.run("rm -r docs/_build")


@task
def docs_dev(ctx: Context):
    """Automatically rebuild the documentation when changes are detected."""
    ctx.run("python -m sphinx_autobuild -b html docs docs/_build/html --open-browser")


@task
def download_moa(ctx: Context):
    """Download moa.jar from the web."""
    url = ctx["moa_url"]
    moa_path = Path(ctx["moa_path"])
    if not moa_path.exists():
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

    if all_exist(directories=["src/moa-stubs", "src/com-stubs/yahoo/labs/samoa"]):
        print("Nothing todo: Java stubs already exist.")
        return

    ctx.run(
        "python -m stubgenj "
        f"--classpath {class_path} "
        "--output-dir src "
        "--convert-strings --no-jpackage-stubs "
        "moa com.yahoo.labs.samoa"
    )


@task
def clean_stubs(ctx: Context):
    """Remove the Java stubs."""
    ctx.run("rm -r src/moa-stubs src/com-stubs")


@task
def clean_moa(ctx: Context):
    """Remove the moa.jar file."""
    moa_path = Path(ctx["moa_path"])
    moa_path.unlink()


@task
def clean(ctx: Context):
    """Clean all build artifacts."""
    clean_stubs(ctx)
    clean_moa(ctx)


@task
def test_notebooks(ctx: Context, parallel: bool = True, overwrite: bool = False):
    """Run the notebooks and check for errors.

    Uses nbmake https://github.com/treebeardtech/nbmake to execute the notebooks and
    check for errors. The `--overwrite` flag can be used to overwrite the notebooks
    with the executed output.
    """

    skip_notebooks = ctx["test_skip_notebooks"]
    print(f"Skipping notebooks: {skip_notebooks}")
    cmd = [
        "python -m pytest --nbmake",
        "-x",  # Stop after the first failure
        "--nbmake-timeout=500",
        "notebooks",
        "--durations=0",  # Show the duration of each notebook
    ]
    cmd += ["-n=auto"] if parallel else []  # Should we run in parallel?
    cmd += (
        ["--overwrite"] if overwrite else []
    )  # Overwrite the notebooks with the executed output
    cmd += ["--deselect " + nb for nb in skip_notebooks]  # Skip some notebooks
    ctx.run(" ".join(cmd))


@task
def unittest(ctx: Context, parallel: bool = True):
    """Run the tests using pytest."""
    cmd = ["python -m pytest", "--durations=0"]  # Show the duration of each test
    cmd += ["-n=auto"] if parallel else []
    ctx.run(" ".join(cmd))


@task
def all_tests(ctx: Context, parallel: bool = True):
    """Run all the tests."""
    test_notebooks(ctx, parallel)
    unittest(ctx, parallel)


docs = Collection("docs")
docs.add_task(docs_build, "build")
docs.add_task(docs_clean, "clean")
docs.add_task(docs_dev, "dev")

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
