# SPDX-License-Identifier: BSD-3-Clause
import subprocess
import platform
import os
from openmoa._prepare_jpype import _get_java_home
import tempfile
from pathlib import Path
from openmoa.env import openmoa_moa_jar
import pytest
import shutil

PYTHON_EXE = os.sys.executable
CMD = [PYTHON_EXE, "-c", "import openmoa"]
CMD_ABOUT = [PYTHON_EXE, "-c", "import openmoa; openmoa.about()"]


def _decode_output(data: bytes) -> str:
    return data.decode(errors="replace")


@pytest.fixture
def env():
    return os.environ.copy()


def test_bad_infer_java_home(env):
    """Tests reporting errors when java cannot be found."""
    env.pop("JAVA_HOME", None)
    env["PATH"] = ""
    assert "JAVA_HOME" not in env
    result = subprocess.run(CMD_ABOUT, capture_output=True, env=env)
    print(_decode_output(result.stdout))
    assert result.returncode != 0
    exception = _decode_output(result.stderr).splitlines()[-1]
    assert exception == (
        "openmoa._prepare_jpype.OpenmoaImportError: Java not found ensure "
        "`java -version` runs successfully. Alternatively, you may set the "
        "JAVA_HOME environment variable to the path of your Java installation "
        "for non-standard installations."
    )


def test_good_java_home(env):
    env["JAVA_HOME"] = _get_java_home().as_posix()
    result = subprocess.run(CMD, capture_output=True, env=env)
    assert result.returncode == 0


def test_bad_java_home(env):
    notfound = Path("/notfound")
    env["JAVA_HOME"] = notfound.as_posix()
    result = subprocess.run(CMD_ABOUT, capture_output=True, env=env)
    assert result.returncode != 0
    exception = _decode_output(result.stderr).splitlines()[-1]
    assert exception == (
        f"openmoa._prepare_jpype.OpenmoaImportError: The JAVA_HOME (`{str(notfound)}`) "
        "environment variable is set, but the path does not exist."
    )


def test_openmoa_moa_jar(env):
    notfound = Path("/notfound")
    env["OPENMOA_MOA_JAR"] = notfound.as_posix()
    result = subprocess.run(CMD_ABOUT, capture_output=True, env=env)
    assert result.returncode != 0
    exception = _decode_output(result.stderr).splitlines()[-1]
    assert exception == (
        f"openmoa._prepare_jpype.OpenmoaImportError: MOA jar not found at `{str(notfound)}`."
    )


def test_nonascii_openmoa(env):
    """Jpype and java used to struggle to start if the path to Jars contains
    non-ascii characters. This test ensures that this is no longer an issue.
    """
    if platform.system() == "Windows":
        pytest.skip("Investigate why this fails on Windows and fix it.")

    with tempfile.TemporaryDirectory(suffix="☺") as d:
        moa_jar = shutil.copyfile(openmoa_moa_jar(), Path(d) / "moa.jar")
        env["OPENMOA_MOA_JAR"] = moa_jar.as_posix()
        result = subprocess.run(
            [
                PYTHON_EXE,
                "-c",
                "from openmoa.env import openmoa_moa_jar; print(openmoa_moa_jar())",
            ],
            capture_output=True,
            env=env,
        )
        assert result.returncode == 0
        assert _decode_output(result.stdout).splitlines()[-1].strip() == moa_jar.as_posix()


def test_openmoa_datasets_dir(env):
    with tempfile.TemporaryDirectory() as d:
        env["OPENMOA_DATASETS_DIR"] = d
        result = subprocess.run(CMD_ABOUT, capture_output=True, env=env)
        assert result.returncode == 0
        about = _decode_output(result.stdout)
        assert f"OPENMOA_DATASETS_DIR: {d}" in about


def test_openmoa_jvm_args(env):
    env["OPENMOA_JVM_ARGS"] = "-Xmx16g -Xss10M"
    result = subprocess.run(CMD_ABOUT, capture_output=True, env=env)
    assert result.returncode == 0
    about = _decode_output(result.stdout)
    assert "OPENMOA_JVM_ARGS:     ['-Xmx16g', '-Xss10M']" in about