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

   .. tab-set::

      .. tab-item:: uv (recommended)

         .. code-block:: bash

            cd CapyMOA
            uv sync --extra dev --extra doc --extra torch-cpu

         ``--extra torch-cpu`` installs CPU-only PyTorch wheels, resolved
         automatically via the ``pytorch-cpu`` index configured in
         ``pyproject.toml`` -- no manual ``--index-url`` step needed. If you
         have a GPU and want CUDA-enabled PyTorch instead, use
         ``--extra torch`` in place of ``--extra torch-cpu`` (the two are
         mutually exclusive).

      .. tab-item:: pip / conda

         .. code-block:: bash

            cd CapyMOA
            pip install --editable ".[dev,doc,torch]"

         The ``dev`` extra does not include ``torch`` -- add it explicitly (as
         above) to run the whole test suite. On Linux, install the CPU build of
         PyTorch first if you do not want the CUDA packages -- see the PyTorch
         note in :doc:`/setup/index`.


#. **Congratulations!**

   You have successfully installed CapyMOA in editable mode.

   A number of utility scripts are defined in ``tasks.py`` to perform common
   tasks. You can list all available tasks by running:

   .. tab-set::

      .. tab-item:: uv

         .. code-block:: bash

            uv run invoke --list

      .. tab-item:: pip / conda (activated venv)

         .. code-block:: bash

            python -m invoke --list # or `invoke --list`

   .. program-output:: python -m invoke --list

   Each of these tasks can be run in the terminal through ``invoke <task>``.
   If you installed with ``uv sync`` (and have not separately activated
   ``.venv``), prefix every invocation with ``uv run`` instead --
   ``uv sync`` does not put ``.venv`` on ``PATH`` the way activating it does.
   Running the task to build documentation would look like this:

   .. tab-set::

      .. tab-item:: uv

         .. code-block:: bash

            uv run invoke docs.build

      .. tab-item:: pip / conda (activated venv)

         .. code-block:: bash

            invoke docs.build

   See the :doc:`/contributing/index` guide for more information on how to
   contribute to CapyMOA.

Testing PyTorch-Optional Code
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

PyTorch is an optional extra (see the PyTorch note in :doc:`/setup/index`), so
code that touches it needs to be tested on both sides of that boundary. Which
pattern to reach for depends on what you're asserting, all illustrated in
``tests/no_torch/test_no_torch.py``:

* **"Needs torch installed"** -- ``pytest.importorskip("torch")`` at the top
  of the test. Use this for anything that only makes sense with the extra
  present; it skips cleanly in a torch-free run rather than failing.

* **"Must work without torch, even though torch happens to be installed"**
  -- the ``run_without_torch`` fixture (``tests/no_torch/conftest.py``), which
  runs a code snippet in a subprocess with ``torch`` blocked on
  ``sys.meta_path``. This is what most torch-optional tests want: it runs in
  the normal CI matrix and dev machines without needing a separate
  environment, so it catches a stray top-level ``import torch`` immediately.

* **"Needs torch to be genuinely absent from the environment"** -- a plain
  ``@pytest.mark.skipif(importlib.util.find_spec("torch") is not None, ...)``.
  Reserve this for the rare assertion the subprocess simulation can't make
  (e.g. that torch was never pulled onto disk in the first place). It only
  runs for real in the CI job that installs with no extras.

Any test using the second or third pattern belongs under ``tests/no_torch/``:
CI points at that whole directory when checking the no-extras install, so a
new file there is picked up automatically -- no workflow changes needed.