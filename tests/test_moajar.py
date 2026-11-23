from hashlib import sha256
from pathlib import Path

import jpype

import capymoa

_MOA_JAR_HASH = "f758bf22dfbf4e37e2ad6094d52dc6943ae10ff8e1e046adbf741388881ea092"


def test_imports() -> None:
    assert capymoa
    """Test that the correct moa version is being packaged"""
    assert jpype.isJVMStarted(), (
        "JVM should be started automatically when importing capymoa"
    )
    jar_path = Path(jpype.getClassPath())
    assert jar_path.suffix == ".jar", "MOA jar should be in the class path"

    with open(jar_path, "rb") as f:
        jar_hash = sha256(f.read()).hexdigest()

    assert jar_hash == _MOA_JAR_HASH, (
        "MOA jar hash should match the expected hash. "
        "Try `invoke refresh-moa` to download the correct version. "
        "If you are expecting a new version update the `_MOA_JAR_HASH` variable`"
    )
