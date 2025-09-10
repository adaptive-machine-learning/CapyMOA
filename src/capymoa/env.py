"""

CapyMOA supports a few environment variables that can be used to customize its behavior.
None of these are required, but they can be useful in certain situations.

* Use ``CAPYMOA_DATASETS_DIR`` to specify a custom directory where datasets will be stored.
    (See :func:`capymoa_datasets_dir`)
* Use ``CAPYMOA_JVM_ARGS`` to specify custom JVM arguments.
    (See :func:`capymoa_jvm_args`)
* Use ``CAPYMOA_MOA_JAR`` to specify a custom MOA jar file.
    (See :func:`capymoa_moa_jar`)
* Use ``JAVA_HOME`` to specify the path to your Java installation.
"""

from os import environ
from pathlib import Path
from typing import List


def capymoa_datasets_dir() -> Path:
    """Return the ``CAPYMOA_DATASETS_DIR`` environment variable or the default value ``./data``.


    The ``CAPYMOA_DATASETS_DIR`` environment variable can be used to specify a custom
    directory where datasets will be stored. Set it to a custom value in bash like this:

    ..  code-block:: bash

        export CAPYMOA_DATASETS_DIR=/path/to/datasets
        python my_capy_moa_script.py

    We recommend setting this environment variable in your shell configuration file.

    :return: The path to the datasets directory.
    """
    dataset_dir = Path(environ.get("CAPYMOA_DATASETS_DIR", "./data"))
    dataset_dir.mkdir(exist_ok=True)
    return dataset_dir


def capymoa_jvm_args() -> List[str]:
    """Return the ``CAPYMOA_JVM_ARGS`` environment variable or the default value ``-Xmx8g -Xss10M``.

    The ``CAPYMOA_JVM_ARGS`` environment variable can be used to specify custom JVM
    arguments. Set it to a custom value in bash like this:

    ..  code-block:: bash

        export CAPYMOA_JVM_ARGS="-Xmx16g -Xss10M"
        python my_capy_moa_script.py

    :return: A list of JVM arguments.
    """
    return environ.get("CAPYMOA_JVM_ARGS", "-Xmx8g -Xss10M").split()


def capymoa_moa_jar() -> Path:
    """Return the ``CAPYMOA_MOA_JAR`` environment variable or the built-in MOA jar file.

    **This is an advanced feature that is unnecessary for most users.**

    The ``CAPYMOA_MOA_JAR`` environment variable can be used to specify a custom path to
    the MOA jar file. Set it to a custom value in bash like this:

    ..  code-block:: bash

        export CAPYMOA_MOA_JAR=/path/to/moa.jar
        python my_capy_moa_script.py

    :return: The path to the MOA jar file.
    """
    default_moa_jar = Path(__file__).parent / "jar" / "moa.jar"
    return Path(environ.get("CAPYMOA_MOA_JAR", default_moa_jar))
