import jpype
from pathlib import Path
from hashlib import sha256
import capymoa

_MOA_JAR_HASH = "fa5fd4fcc4dad46734149f97b69a309bd2ff57d1f6227879f40cce239cf6be35"


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
