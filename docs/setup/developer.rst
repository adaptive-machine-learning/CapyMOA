Developer Setup
---------------

If you want to make changes to CapyMOA, you should follow these steps to set up
an editable installation of CapyMOA, with development and documentation
dependencies.

#. **Dependencies**

   Follow the instructions above to install PyTorch, Java, and optionally a
   virtual environment.

#. **Pandoc** 
   
   Ensure that you have `Pandoc <https://pandoc.org/>`__ installed on your system.
   If it's not installed, you can install it by running the following command on

   .. tab-set::

      .. tab-item:: Ubuntu

         .. code-block:: bash

               sudo apt-get install -y pandoc

      .. tab-item:: macOS

         .. code-block:: bash

               sudo brew install pandoc

      .. tab-item:: Windows/Other

         Follow the instructions on the `Pandoc website <https://pandoc.org/installing.html>`__.

      .. tab-item:: conda

         .. code-block:: bash

               conda install -c conda-forge pandoc


#. **Clone the Repository**
   
   If you want to contribute to CapyMOA, you should clone the repository,
   install development dependencies, and install CapyMOA in editable mode.

   If you are intending to contribute to CapyMOA, consider making a
   `fork <https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/working-with-forks/fork-a-repo>`__
   of the repository and cloning your fork instead of the main
   repository. This way, you can push changes to your fork and create
   pull requests to the main repository.

   .. code:: bash

      git clone https://github.com/adaptive-machine-learning/CapyMOA.git
      # or clone via the SSH protocol (often preferred if you use SSH keys for git):
      #   ``git clone with git@github.com:adaptive-machine-learning/CapyMOA.git``
      

#. **Install CapyMOA in Editable Mode**

   To install CapyMOA in editable mode with development and documentation
   dependencies, navigate to the root of the repository and run:

   .. code-block:: bash

      cd CapyMOA
      pip install --editable ".[dev,doc]"


#. **Congratulations!**

   You have successfully installed CapyMOA in editable mode.

   A number of utility scripts are defined in ``tasks.py`` to perform common
   tasks. You can list all available tasks by running:

   .. code-block:: bash

      python -m invoke --list # or `invoke --list`

   .. program-output:: python -m invoke --list

   Each of these tasks can be run in the terminal through ``invoke <task>``. Running the task to build documentation would look like this:

   .. code-block:: bash

      invoke docs.build

   See the :doc:`/contributing/index` guide for more information on how to
   contribute to CapyMOA.