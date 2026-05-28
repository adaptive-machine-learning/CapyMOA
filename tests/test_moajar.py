# SPDX-License-Identifier: BSD-3-Clause
import jpype
from pathlib import Path
from hashlib import sha256
import openmoa
from openmoa._prepare_jpype import _start_jpype

_MOA_JAR_HASH = "b14be3c1df87aa5bf37f24c9a35258ab1f9a941897e61294701c43c0141dc2b7"


def test_imports() -> None:
    """Test that the correct MOA version is being packaged."""
    assert openmoa
    _start_jpype()
    assert jpype.isJVMStarted(), (
        "JVM should be started before checking the packaged MOA jar"
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