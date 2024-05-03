import capymoa
import jpype
from pathlib import Path
from hashlib import sha256
import warnings
import os

_MOA_JAR_HASH="725761e4ea28fea96b9f33847042133e41c33d4451f28a70bdd856d438b0b925"

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

