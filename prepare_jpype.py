# Use this to check if java_home is correctly set
import subprocess
import configparser
import jpype
import jpype.imports
from jpype.types import *

def start_jpype():
    # Create a configuration parser
    config = configparser.ConfigParser()
    config.read('config.ini')
    # Get the MOA JAR path from the configuration file
    moa_jar_path = config['Paths']['moa_jar_path']

    # Add the moa jar to the class path
    jpype.addClassPath(moa_jar_path)

    print(f"MOA jar path location (config.ini): {moa_jar_path}")
    print("JVM Location (system): ")
    subprocess.call("ECHO $JAVA_HOME", shell=True)
    # If JAVA_HOME is not set, then jpype will fail. 

    if not jpype.isJVMStarted():
        jpype.startJVM()
        # Add the moa jar to the class path
        jpype.addClassPath(moa_jar_path)
        print("Sucessfully started the JVM and added MOA jar to the class path")
    else:
        print("JVM already started")

# The JVM automatically shutdown with python, no need to explicitly call the shutdown method
# https://jpype.readthedocs.io/en/latest/userguide.html#shutdownjvm

if __name__ == "__main__":
    start_jpype()