import subprocess
import platform
import os
from capymoa._prepare_jpype import _get_java_home
import tempfile
from pathlib import Path
from capymoa.env import capymoa_moa_jar
import pytest
import shutil

PYTHON_EXE = os.sys.executable
CMD = [PYTHON_EXE, "-c", "import capymoa"]
CMD_ABOUT = [PYTHON_EXE, "-c", "import capymoa; capymoa.about()"]


@pytest.fixture
def env():
    return os.environ.copy()


def test_bad_infer_java_home(env):
    """Tests reporting errors when java cannot be found."""
    del env["JAVA_HOME"]
    env["PATH"] = ""
    assert "JAVA_HOME" not in env
    result = subprocess.run(CMD_ABOUT, capture_output=True, env=env)
    print(result.stdout.decode())
    assert result.returncode != 0
    exception = result.stderr.decode().splitlines()[-1]
    assert exception == (
        "capymoa._prepare_jpype.CapymoaImportError: Java not found ensure "
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
    result = subprocess.run(CMD, capture_output=True, env=env)
    assert result.returncode != 0
    exception = result.stderr.decode().splitlines()[-1]
    assert exception == (
        f"capymoa._prepare_jpype.CapymoaImportError: The JAVA_HOME (`{str(notfound)}`) "
        "environment variable is set, but the path does not exist."
    )


def test_capymoa_moa_jar(env):
    notfound = Path("/notfound")
    env["CAPYMOA_MOA_JAR"] = notfound.as_posix()
    result = subprocess.run(CMD, capture_output=True, env=env)
    assert result.returncode != 0
    exception = result.stderr.decode().splitlines()[-1]
    assert exception == (
        f"capymoa._prepare_jpype.CapymoaImportError: MOA jar not found at `{str(notfound)}`."
    )


def test_nonascii_capymoa(env):
    """Jpype and java used to struggle to start if the path to Jars contains
    non-ascii characters. This test ensures that this is no longer an issue.
    """
    if platform.system() == "Windows":
        pytest.skip("Investigate why this fails on Windows and fix it.")

    with tempfile.TemporaryDirectory(suffix="☺") as d:
        moa_jar = shutil.copyfile(capymoa_moa_jar(), Path(d) / "moa.jar")
        env["CAPYMOA_MOA_JAR"] = moa_jar.as_posix()
        result = subprocess.run(
            [
                PYTHON_EXE,
                "-c",
                "from capymoa.env import capymoa_moa_jar; print(capymoa_moa_jar())",
            ],
            capture_output=True,
            env=env,
        )
        assert result.returncode == 0
        assert result.stdout.decode().splitlines()[-1].strip() == moa_jar.as_posix()


def test_capymoa_datasets_dir(env):
    with tempfile.TemporaryDirectory() as d:
        env["CAPYMOA_DATASETS_DIR"] = d
        result = subprocess.run(CMD_ABOUT, capture_output=True, env=env)
        assert result.returncode == 0
        about = result.stdout.decode()
        assert f"CAPYMOA_DATASETS_DIR: {d}" in about


def test_capymoa_jvm_args(env):
    env["CAPYMOA_JVM_ARGS"] = "-Xmx16g -Xss10M"
    result = subprocess.run(CMD_ABOUT, capture_output=True, env=env)
    assert result.returncode == 0
    about = result.stdout.decode()
    assert "CAPYMOA_JVM_ARGS:     ['-Xmx16g', '-Xss10M']" in about
