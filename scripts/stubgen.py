"""Uses https://pypi.org/project/stubgenj/ https://gitlab.cern.ch/scripting-tools/stubgenj
to generate Python stubs for Java classes. This is useful for type checking and
auto-completion in IDEs. The generated stubs are placed in the `src` directory
with the `-stubs` suffix. This script is intended to be run from the project root
as part of the build process.
"""
import importlib
import logging
from pathlib import Path
from typing import List

import jpype.imports
from stubgenj import generateJavaStubs

log = logging.getLogger(__name__)


def do_stubs_exist(java_prefixes: List[str], output_dir: str):
    output_dir = Path(output_dir)
    for prefix in java_prefixes:
        prefix = prefix.split(".")[0]
        if not (output_dir / f"{prefix}-stubs").exists():
            return False
    return True


def main():
    logging.basicConfig(level="INFO")
    classpath = ["src/capymoa/jar/moa.jar"]
    prefixes = [
        "moa",
        "com.yahoo.labs.samoa",
    ]
    output_dir = "src"
    convert_strings = True

    # Avoid rerunning if the stubs are already generated
    if do_stubs_exist(prefixes, output_dir):
        log.info("Stubs already exist. Skipping generation.")
        log.info(
            "To regenerate the stubs, delete the existing stubs and rerun the script."
        )
        return

    log.info("Starting JPype JVM with classpath " + str(classpath))
    jpype.startJVM(classpath=classpath, convertStrings=convert_strings)  # noqa: exists
    prefixPackages = [importlib.import_module(prefix) for prefix in prefixes]
    generateJavaStubs(
        prefixPackages,
        outputDir=output_dir,
        # Add '-stubs' to the package name following PEP-561
        useStubsSuffix=True,
        # Do not create a partial jpype-stubs
        jpypeJPackageStubs=False,
        # Generate docstrings from JavaDoc where available
        includeJavadoc=True,
    )
    log.info("Generation done.")
    jpype.java.lang.Runtime.getRuntime().halt(0)


if __name__ == "__main__":
    assert (Path.cwd() / ".git").exists(), "Run this script from the project root."
    main()
