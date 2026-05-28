# SPDX-License-Identifier: BSD-3-Clause
# Python imports
import jpype
import jpype.imports
import os
from pathlib import Path
from hashlib import sha256
import shutil
import subprocess
from .__about__ import __version__
from .env import (
    openmoa_jvm_args,
    openmoa_moa_jar,
    openmoa_datasets_dir,
)

_OPENMOA_PACKAGE_ROOT = Path(__file__).parent


class OpenmoaImportError(RuntimeError):
    pass


def _get_java_home() -> Path:
    """Find java home.

    Respects the JAVA_HOME environment variable if it is set, otherwise tries to
    find the java home by running a special java program that prints it.
    """

    if "JAVA_HOME" in os.environ:
        java_home = Path(os.environ["JAVA_HOME"])

        if not java_home.exists():
            raise OpenmoaImportError(
                f"The JAVA_HOME (`{java_home}`) environment variable is set, "
                "but the path does not exist."
            )
    else:
        # We can find the java home by asking a special java program to print it for us
        java_class_path = _OPENMOA_PACKAGE_ROOT / "jar"
        try:
            result = subprocess.run(
                ["java", "-classpath", java_class_path.as_posix(), "Home"],
                capture_output=True,
            )
        except FileNotFoundError:
            raise OpenmoaImportError(
                "Java not found ensure `java -version` runs successfully. "
                "Alternatively, you may set the JAVA_HOME environment variable to the "
                "path of your Java installation for non-standard installations."
            )

        java_home = Path(result.stdout.decode().strip())

        assert java_home.exists(), (
            f"The java.home reported by the java program does not exist: {java_home}"
        )

    return java_home



def _classpath_moa_jar(moa_jar: Path) -> Path:
    """Return a JVM classpath-safe copy of the MOA jar when needed.

    JPype's package import hook can fail on Windows for jar paths under
    directories whose names contain dashes. A cached copy under the system temp
    directory avoids that path-shape issue without changing the packaged jar.
    """
    if os.name != "nt":
        return moa_jar

    with open(moa_jar, "rb") as f:
        jar_hash = sha256(f.read()).hexdigest()
    cache_dir = Path.home() / ".openmoa" / "moa" / jar_hash
    cache_dir.mkdir(parents=True, exist_ok=True)
    cached_jar = cache_dir / "moa.jar"
    if not cached_jar.exists() or cached_jar.stat().st_size != moa_jar.stat().st_size:
        shutil.copyfile(moa_jar, cached_jar)
    return cached_jar


def _moa_hash():
    with open(openmoa_moa_jar(), "rb") as f:
        return sha256(f.read()).hexdigest()


def about():
    """Print useful debug information about the OpenMOA setup.

    >>> import openmoa
    >>> openmoa.about() # doctest: +ELLIPSIS
    OpenMOA ...
    """
    _start_jpype()
    java_version = jpype.java.lang.System.getProperty("java.version")
    print(f"OpenMOA {__version__}")
    print(f"  OPENMOA_DATASETS_DIR: {openmoa_datasets_dir()}")
    print(f"  OPENMOA_MOA_JAR:      {openmoa_moa_jar()}")
    print(f"  OPENMOA_JVM_ARGS:     {openmoa_jvm_args()}")
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
    moa_jar = openmoa_moa_jar()
    if not (moa_jar.exists() and moa_jar.is_file()):
        raise OpenmoaImportError(f"MOA jar not found at `{moa_jar}`.")
    jpype.addClassPath(_classpath_moa_jar(moa_jar).as_posix())

    # Start the JVM
    jpype.startJVM(jpype.getDefaultJVMPath(), *openmoa_jvm_args())

    # The JVM automatically shutdown with python, no need to explicitly call the shutdown method
    # https://jpype.readthedocs.io/en/latest/userguide.html#shutdownjvm
