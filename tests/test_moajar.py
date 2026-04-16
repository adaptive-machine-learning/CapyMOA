import jpype
from pathlib import Path
from hashlib import sha256
import capymoa

_MOA_JAR_HASH = "e2e5661d361c7e42912be4546168b10e858e85e64a75bc8627d5a5493ec30e63"


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
