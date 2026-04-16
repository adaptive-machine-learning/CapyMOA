.. toctree::
   :hidden:

   docker
   developer

.. _setup:

Setup
=====

This document describes how to install CapyMOA and its dependencies. CapyMOA is
tested against Python 3.10, 3.11, and 3.12. Newer versions of Python will likely
work but have yet to be tested.

Once you have installed the :ref:`dependencies`, you may
install CapyMOA using pip (optionally in a :ref:`venv`):

.. code:: bash

   pip install capymoa

To verify your installation, run:

.. code:: bash

   python -c "import capymoa; print(capymoa.__version__)"

.. _venv:

Virtual Environment
^^^^^^^^^^^^^^^^^^^

We recommend using a virtual environment to isolate CapyMOA and its dependencies
from your other projects. This is especially important if you have other
projects that require different versions of the same dependencies.

If you chose to use a virtual environment, you have some choices:

*  **Python Virtual Environment**
   PyVenv is a built-in tool for creating virtual
   environments in Python. You can create a new virtual environment with:

   .. code:: bash

      python3 -m venv .capymoa-venv
      source .capymoa-venv/bin/activate
      # On Windows, use `.capymoa-venv\Scripts\activate`

*  **Conda Environment**
   Miniconda is a good choice for managing Python environments. You can install
   Miniconda from `here <https://docs.conda.io/en/latest/miniconda.html>`__.
   Once you have Miniconda installed, you can create a new environment with:

   .. code:: bash

      conda create -n capymoa python=3.11
      conda activate capymoa

   When your environment is activated, you can install CapyMOA by following the
   instructions below.

.. _dependencies:

Dependencies
^^^^^^^^^^^^

CapyMOA has some required dependencies that may require manual installation
before CapyMOA can be used:

Java
~~~~

CapyMOA requires a Java runtime. You can check if Java is installed by running
the following command in your terminal:

.. code:: bash

   java -version

If Java is not installed, you can download OpenJDK (Open Java Development
Kit) from `this link <https://openjdk.org/install/>`__, or alternatively the
Oracle JDK from `this link <https://www.oracle.com/java>`__.  You only need
to install the Java Runtime (JRE). Linux and macOS users can also install
OpenJDK using their distribution's package manager:

.. tab-set::

   .. tab-item:: Ubuntu

      .. code-block:: bash

            sudo apt-get install -y default-jre-headless

   .. tab-item:: macOS

      .. code-block:: bash

            brew install openjdk

CapyMOA will attempt to find the Java automatically unless the ``JAVA_HOME``
environment variable is set. This allows you to have multiple Java versions
or have Java installed outside of the system path.

PyTorch
~~~~~~~

The CapyMOA algorithms using deep learning require PyTorch. If you want to use
these algorithms, follow the instructions
`here <https://pytorch.org/get-started/locally/>`__ to get the correct version for
your hardware. Ensure that you install PyTorch in the same virtual environment
where you want to install CapyMOA.

For CPU only, you can install PyTorch with:

.. code:: bash

   pip3 install torch torchvision torchaudio \
      --index-url https://download.pytorch.org/whl/cpu

