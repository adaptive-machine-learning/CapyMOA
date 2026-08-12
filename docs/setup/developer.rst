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
code that touches it needs to be tested on both sides of that boundary. CI
runs the suite twice: once with ``-m "not torch"`` and no PyTorch installed,
once with ``-m "torch"`` and ``--extra torch-cpu``.

Mark any test that needs torch with ``@pytest.mark.torch`` (or
``pytest.param(..., marks=pytest.mark.torch)`` for one case of a
parametrized test), so ``-m "not torch"`` deselects it.

That alone isn't enough for a module that needs torch just to be
*collected* -- e.g. one that imports ``capymoa.ocl`` at module scope --
since collection happens before marker-based deselection. For those, call
``pytest.markskip("torch")`` (defined in ``tests/conftest.py``) before the
torch-touching import:

.. code-block:: python

   import pytest

   pytest.markskip("torch")

   import torch
   from capymoa.ocl.util._buffer_list import BufferList

   pytestmark = pytest.mark.torch

``markskip`` only checks whether ``-m`` selects the mark, not whether the
dependency is actually importable, so it's safe to rely on at module level:
when torch tests are wanted (``-m "torch"``, or no ``-m`` filter at all) but
torch isn't installed, the import right after ``markskip`` fails loudly
instead of being silently skipped.

For a file that mixes torch and non-torch cases, call ``markskip`` lazily
inside the constructor for just the torch-only case instead of guarding the
whole module -- see ``_make_finetune`` in ``tests/test_classifiers.py``.