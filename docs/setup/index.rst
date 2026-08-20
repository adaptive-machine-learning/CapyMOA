.. toctree::
   :hidden:

   docker
   developer

.. _setup:

Setup
=====

This document describes how to install CapyMOA and its dependencies. CapyMOA is
tested against Python 3.11, 3.12, and 3.13. Newer versions of Python will likely
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

*  **uv**
   `uv <https://docs.astral.sh/uv/>`__ is a fast Python package and project
   manager. You can create a new virtual environment with:

   .. code:: bash

      uv venv .capymoa-venv
      source .capymoa-venv/bin/activate
      # On Windows, use `.capymoa-venv\Scripts\activate`

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

PyTorch is **optional**. ``pip install capymoa`` does not install it, so the
core of CapyMOA -- streams, classifiers, regressors, drift detectors and
evaluation -- installs without pulling a deep-learning stack.

The parts of CapyMOA that use deep learning do require it:
:mod:`capymoa.ocl`, :class:`capymoa.stream.TorchStream`,
the ``Batch*`` learners, :class:`capymoa.anomaly.Autoencoder`,
:class:`capymoa.classifier.Finetune` and :class:`capymoa.ssl.OSNN`. Using one of
those without PyTorch raises an ``OptionalDependencyError`` telling you what to
install.

:class:`capymoa.drift.detectors.ABCD` is a partial case: it works without
PyTorch on its default ``model_id="pca"`` and on ``"kpca"``, and needs the extra
only for the autoencoder model, ``model_id="ae"``.

Install CapyMOA with PyTorch using the ``torch`` extra:

.. code:: bash

   pip install capymoa[torch]

.. note::

   On Linux the default PyPI PyTorch wheel is CUDA-enabled and pulls the NVIDIA
   stack (several GB). If you do not need a GPU, install the CPU build *first*
   and then CapyMOA -- pip (or uv) will keep the version you already have:

   .. tab-set::

      .. tab-item:: pip

         .. code-block:: bash

            pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
            pip install capymoa[torch]

      .. tab-item:: uv

         .. code-block:: bash

            uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
            uv pip install capymoa[torch]

To match a specific GPU or CUDA version instead, follow the instructions
`here <https://pytorch.org/get-started/locally/>`__, and make sure PyTorch goes
into the same virtual environment as CapyMOA.

