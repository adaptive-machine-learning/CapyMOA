# Python imports
import subprocess
import configparser
import jpype
import jpype.imports
from jpype.types import *
import os
from pathlib import Path


def _start_jpype():
    capymoa_root = Path(__file__).resolve().parent
    print(f"capymoa_root: {capymoa_root}")

    # Create a configuration parser
    config = configparser.ConfigParser()
    config.read(capymoa_root / "config.ini")

    # Obtain the MOA JAR path and JVM args from the configuration file
    moa_jar_path = config["Paths"]["moa_jar_path"]
    jvm_args = config["JVM"]["args"].split(" ")

    moa_jar = Path(moa_jar_path)
    if not moa_jar.is_absolute():
        moa_jar = capymoa_root / moa_jar_path
    if not moa_jar.exists():
        raise FileNotFoundError(f"MOA jar not found at {moa_jar}")

    # Add the moa jar to the class path
    jpype.addClassPath(moa_jar)

    # If JAVA_HOME is not set, then jpype will fail.
    if not jpype.isJVMStarted():
        print(f"MOA jar path location (config.ini): {moa_jar}")
        print("JVM Location (system): ")
        print(f"JAVA_HOME: {os.environ.get('JAVA_HOME', 'Inferred from system')}")
        print(f"JVM args: {jvm_args}")

        jpype.startJVM(jpype.getDefaultJVMPath(), *jvm_args)
        # Add the moa jar to the class path
        # jpype.addClassPath(moa_jar_path)
        print("Sucessfully started the JVM and added MOA jar to the class path")
    # else:
    #     print("JVM already started")


# The JVM automatically shutdown with python, no need to explicitly call the shutdown method
# https://jpype.readthedocs.io/en/latest/userguide.html#shutdownjvm
