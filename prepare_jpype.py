# Python imports
import subprocess
import configparser
import jpype
import jpype.imports
from jpype.types import *
import os


def start_jpype():
    # Create a configuration parser
    config = configparser.ConfigParser()
    config.read("config.ini")
    # Obtain the MOA JAR path and JVM args from the configuration file
    moa_jar_path = config["Paths"]["moa_jar_path"]
    jvm_args = config["JVM"]["args"].split(" ")

    # Add the moa jar to the class path
    jpype.addClassPath(moa_jar_path)

    # If JAVA_HOME is not set, then jpype will fail.
    if not jpype.isJVMStarted():
        print(f"MOA jar path location (config.ini): {moa_jar_path}")
        print("JVM Location (system): ")
        subprocess.call("ECHO $JAVA_HOME", shell=True)

        print(f"JVM args: {jvm_args}")
        jpype.startJVM(jpype.getDefaultJVMPath(), *jvm_args)
        # Add the moa jar to the class path
        jpype.addClassPath(moa_jar_path)
        print("Sucessfully started the JVM and added MOA jar to the class path")
    # else:
    #     print("JVM already started")


# The JVM automatically shutdown with python, no need to explicitly call the shutdown method
# https://jpype.readthedocs.io/en/latest/userguide.html#shutdownjvm

if __name__ == "__main__":
    start_jpype()
