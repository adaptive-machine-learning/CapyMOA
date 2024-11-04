import jpype
from pathlib import Path
from hashlib import sha256

_MOA_JAR_HASH = "96ad3955de5e702983e611c3eda06df9b2440a39d07524808702be53f5b742b9"


def test_imports() -> None:
    """Test that the correct moa version is being packaged"""
    assert (
        jpype.isJVMStarted()
    ), "JVM should be started automatically when importing capymoa"
    jar_path = Path(jpype.getClassPath())
    assert jar_path.suffix == ".jar", "MOA jar should be in the class path"

    with open(jar_path, "rb") as f:
        jar_hash = sha256(f.read()).hexdigest()

    assert jar_hash == _MOA_JAR_HASH, (
        "MOA jar hash should match the expected hash. "
        "Try `invoke refresh-moa` to download the correct version. "
        "If you are expecting a new version update the `_MOA_JAR_HASH` variable`"
    )
