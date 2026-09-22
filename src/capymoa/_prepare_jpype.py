# Python imports
import os
import subprocess
from hashlib import sha256
from pathlib import Path

import jpype
import jpype.imports

from .__about__ import __version__
from .env import (
    capymoa_datasets_dir,
    capymoa_jvm_args,
    capymoa_moa_jar,
)

_CAPYMOA_PACKAGE_ROOT = Path(__file__).parent


class CapymoaImportError(RuntimeError):
    pass


def _get_java_home() -> Path:
    """Find java home.

    Respects the JAVA_HOME environment variable if it is set, otherwise tries to
    find the java home by running a special java program that prints it.
    """

    if "JAVA_HOME" in os.environ:
        java_home = Path(os.environ["JAVA_HOME"])

        if not java_home.exists():
            raise CapymoaImportError(
                f"The JAVA_HOME (`{java_home}`) environment variable is set, "
                "but the path does not exist."
            )
    else:
        # We can find the java home by asking a special java program to print it for us
        java_class_path = _CAPYMOA_PACKAGE_ROOT / "jar"
        try:
            result = subprocess.run(
                ["java", "-classpath", java_class_path.as_posix(), "Home"],
                capture_output=True,
                check=True,
            )
        except FileNotFoundError:
            raise CapymoaImportError(
                "Java not found. See https://capymoa.org/setup/#java."
            )
        except subprocess.CalledProcessError as e:
            raise CapymoaImportError(
                f"Failed to determine java.home: {e.stderr.decode().strip()}"
            )

        java_home = Path(result.stdout.decode().strip())

        assert java_home.exists(), (
            f"The java.home reported by the java program does not exist: {java_home}"
        )

    return java_home


def _moa_hash():
    with open(capymoa_moa_jar(), "rb") as f:
        return sha256(f.read()).hexdigest()


def about():
    """Print useful debug information about the CapyMOA setup.

    >>> import capymoa
    >>> capymoa.about() # doctest: +ELLIPSIS
    CapyMOA ...
    """
    java_version = jpype.java.lang.System.getProperty("java.version")
    print(f"CapyMOA {__version__}")
    print(f"  CAPYMOA_DATASETS_DIR: {capymoa_datasets_dir()}")
    print(f"  CAPYMOA_MOA_JAR:      {capymoa_moa_jar()}")
    print(f"  CAPYMOA_JVM_ARGS:     {capymoa_jvm_args()}")
    print(f"  JAVA_HOME:            {_get_java_home()}")
    print(f"  MOA version:          {_moa_hash()}")
    print(f"  JAVA version:         {java_version}")


def _start_jpype():
    # If it has already been started, we don't need to start it again
    if jpype.isJVMStarted():
        return

    # Jpype is looking for the JAVA_HOME environment variable.
    os.environ["JAVA_HOME"] = _get_java_home().as_posix()

    # Add the MOA jar to the classpath
    moa_jar = capymoa_moa_jar()
    if not (moa_jar.exists() and moa_jar.is_file()):
        raise CapymoaImportError(f"MOA jar not found at `{moa_jar}`.")
    jpype.addClassPath(moa_jar)

    # Start the JVM
    args = capymoa_jvm_args()
    # Some versions of Java require this flag to allow access to parts of java.lang.System
    # https://stackoverflow.com/q/79725728
    args.append("--enable-native-access=ALL-UNNAMED")
    # ignoreUnrecognized: on unsupported JVMs (e.g. Java 8), the flag above causes
    # JVM creation to fail with an opaque error before JPype's own clear
    # "Java version too old" check ever runs. Ignoring unrecognized options lets
    # that check surface instead. https://github.com/jpype-project/jpype/issues/1306
    jpype.startJVM(jpype.getDefaultJVMPath(), *args, ignoreUnrecognized=True)

    # The JVM automatically shutdown with python, no need to explicitly call the shutdown method
    # https://jpype.readthedocs.io/en/latest/userguide.html#shutdownjvm
