Installation
============

This document describes how to install OpenMOA and its dependencies. OpenMOA is
tested against Python 3.10, 3.11, 3.12, and 3.13. Newer versions of Python will likely
work but have yet to be tested.

#. **Virtual Environment (Optional)**

   We recommend using a virtual environment to manage your Python
   environment. Miniconda is a good choice for managing Python
   environments. You can install Miniconda from
   `here <https://docs.conda.io/en/latest/miniconda.html>`__. Once you have
   Miniconda installed, you can create a new environment with:

   .. code:: bash

      conda create -n openmoa python=3.11
      conda activate openmoa

   When your environment is activated, you can install OpenMOA by following
   the instructions below.

#. **Java (Required)**

   OpenMOA requires Java to be installed and accessible in your
   environment. You can check if Java is installed by running the following
   command in your terminal:

   .. code:: bash

      java -version

   If Java is not installed, you can download OpenJDK (Open Java
   Development Kit) from `this link <https://openjdk.org/install/>`__, or
   alternatively the Oracle JDK from `this
   link <https://www.oracle.com/java>`__. Linux users can also install
   OpenJDK using their distribution’s package manager.

   Now that Java is installed, you should see an output similar to the
   following when you run the command ``java -version``:

   ::

      openjdk version "17.0.9" 2023-10-17
      OpenJDK Runtime Environment (build 17.0.9+8)
      OpenJDK 64-Bit Server VM (build 17.0.9+8, mixed mode)


#. **PyTorch (Required)**

   The OpenMOA algorithms using deep learning require PyTorch. It is not
   installed by default because different versions are required for
   different hardware. If you want to use these algorithms, follow the
   instructions `here <https://pytorch.org/get-started/locally/>`__ to get
   the correct version for your hardware. Ensure that you install PyTorch in
   the same environment virtual environment where you want to install OpenMOA.

   For CPU only, you can install PyTorch with:

   .. code:: bash

      pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

#. **Install OpenMOA**

   .. code:: bash

      pip install openmoa

   To verify your installation, run:

   .. code:: bash

      python -c "import openmoa; print(openmoa.__version__)"

Install OpenMOA for Development
===============================

If you want to make changes to OpenMOA, you should follow these steps to set up
an editable installation of OpenMOA, with development and documentation
dependencies.

#. **Dependencies**

   Follow the instructions above to install PyTorch, Java, and optionally a
   virtual environment.

#. **Pandoc** 
   
   Ensure that you have `Pandoc <https://pandoc.org/>`_ installed on your system.
   If it's not installed, you can install it by running the following command on

   .. tab-set::

      .. tab-item:: Ubuntu

         .. code-block:: bash

               sudo apt-get install -y pandoc

      .. tab-item:: macOS

         .. code-block:: bash

               sudo brew install pandoc

      .. tab-item:: Window/Other

         Follow the instructions on the `Pandoc website <https://pandoc.org/installing.html>`_.

      .. tab-item:: conda

         .. code-block:: bash

               conda install -c conda-forge pandoc


#. **Clone the Repository**
   
   If you want to contribute to OpenMOA, you should clone the repository,
   install development dependencies, and install OpenMOA in editable mode.

   If you are intending to contribute to OpenMOA, consider making a
   `fork <https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/working-with-forks/fork-a-repo>`__
   of the repository and cloning your fork instead of the main
   repository. This way, you can push changes to your fork and create
   pull requests to the main repository.

   .. code:: bash

      git clone https://github.com/ZW-SIYUAN/OpenMOA.git
      # or clone via the SSH protocol (often preferred if you use SSH keys for git):
      #   ``git clone with git@github.com:ZW-SIYUAN/OpenMOA.git``
      

#. **Install OpenMOA in Editable Mode**

   To install OpenMOA in editable mode with development and documentation
   dependencies, navigate to the root of the repository and run:

   .. code-block:: bash

      cd OpenMOA
      pip install --editable ".[dev,doc]"


#. **Congratulations!**

   You have successfully installed OpenMOA in editable mode.

   A number of utility scripts are defined in ``tasks.py`` to perform common
   tasks. You can list all available tasks by running:

   .. code-block:: bash

      python -m invoke --list # or `invoke --list`

   .. program-output:: python -m invoke --list

   See the :doc:`contributing/index` guide for more information on how to
   contribute to OpenMOA.


