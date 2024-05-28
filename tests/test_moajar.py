import jpype
from pathlib import Path
from hashlib import sha256
import warnings
import os

_MOA_JAR_HASH = "c10abb16c87186f0da887d4a05e755117ce72eb695061735a5148b865979a8c6"


def test_imports():
    assert jpype.isJVMStarted(), "JVM should be started automatically when importing capymoa"
    jar_path = Path(jpype.getClassPath())
    assert jar_path.suffix == ".jar", "MOA jar should be in the class path"

    with open(jar_path, "rb") as f:
        jar_hash = sha256(f.read()).hexdigest()

    if jar_hash != _MOA_JAR_HASH:
        warnings.warn(
            """
            You are using a different version of MOA than what was tested against.
            If you encounter issues with CI/CD this might be the reason.
            """
        )

    if bool(os.environ.get("CI", False)):
        assert jar_hash == _MOA_JAR_HASH, \
            """In the CI/CD _MOA_JAR_HASH must match the hash of the MOA jar in the class path."""

