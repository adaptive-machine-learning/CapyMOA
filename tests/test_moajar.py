from hashlib import sha256
from pathlib import Path

import jpype

import capymoa

_MOA_JAR_HASH = "02988a41827a7c591897218772976a67cba491ff1f1167622ddae7fad4f1df1b"


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
