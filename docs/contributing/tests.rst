Adding Tests
============

Ensure you have installed the development dependencies by following the
instructions in the :doc:`installation guide </setup/developer>`. To run all
tests, use the following command:

.. code-block:: bash

    invoke test

PyTest
------

Tests can be added to the ``tests`` directory. PyTest will automatically
discover and run these tests. They should be named ``test_*.py``, and the
test functions should be named ``test_*``. See the `PyTest documentation
<https://docs.pytest.org>`_ for more information.

Use PyTest style tests for parameterised tests, tests that require fixtures,
and tests that require setup.

These tests can be run with:

.. code-block:: bash

    pytest

Or to run a specific test:

.. code-block:: bash

    pytest tests/test_*.py

Or to run with the same configuration as continuous integration:

.. code-block:: bash

    invoke test.pytest

Testing PyTorch-Optional Code
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

PyTorch is an optional extra (see the PyTorch note in :doc:`/setup/index`).
CI runs the suite twice: once with ``-m "not torch"`` and no PyTorch
installed, once with ``-m "torch"`` and ``--extra torch-cpu``.

* Mark any test that needs torch with ``@pytest.mark.torch``, or
  ``pytest.param(..., marks=pytest.mark.torch)`` for one case of a
  parametrized test, so ``-m "not torch"`` deselects it.
* A module that needs torch just to be *collected* -- e.g. one that imports
  ``capymoa.ocl`` at module scope -- needs more than a marker, since
  collection happens before marker-based deselection. Assign
  ``pytest.markskip("torch")`` (defined in ``tests/conftest.py``) to
  ``pytestmark`` before the torch-touching import:

  .. code-block:: python

      import pytest

      pytestmark = pytest.markskip("torch")

      import torch
      from capymoa.ocl.util._buffer_list import BufferList

  - ``markskip`` returns ``pytest.mark.torch`` when ``-m`` selects it (so
    the assignment marks the whole module, same as
    ``pytestmark = pytest.mark.torch`` written by hand), or raises a
    module-level skip when it doesn't.
  - It never checks whether the dependency is actually importable: when
    torch tests are wanted (``-m "torch"``, or no ``-m`` filter at all) but
    torch isn't installed, the import right after ``markskip`` fails loudly
    instead of being silently skipped.
* For a file that mixes torch and non-torch cases, call ``markskip("torch")``
  bare (discarding the return value) inside the constructor for just the
  torch-only case instead of guarding the whole module -- see
  ``_make_finetune`` in ``tests/test_classifiers.py``.

Doctest
-------

`Doctest <https://docs.python.org/3/library/doctest.html>`_ allows you to
write tests directly in the docstrings of your code, making it easier to
keep documentation up-to-date. The tests are written as examples in a Python
interactive shell.

Use doctest style tests to document code with simple tested examples.

Here's an example of a function with a doctest:

.. code-block:: python

    def hello_world():
        """
        >>> hello_world()
        Hello, World!
        """
        print("Hello, World!")

You can run this test with:

.. code-block:: bash

    pytest --doctest-modules path/to/your/module.py

Alternatively, you can run all unit tests with the same configuration as
continuous integration:

.. code-block:: bash

    invoke test.doctest

Notebooks
---------

We use `nbmake <https://github.com/treebeardtech/nbmake>`_ to test that all
notebooks in the ``notebooks`` directory run without error. This ensures that
the notebooks are always up-to-date and working correctly.

You can run a notebook as a test with:

.. code-block:: bash

    pytest --nbmake notebooks/my_notebook.ipynb

    # Often the examples take too long to run regularly as tests. To speed up
    # testing some notebooks use the NB_FAST environment variable to run the
    # notebook faster by using smaller datasets or fewer iterations. To run
    # them in this mode use:
    NB_FAST=true pytest --nbmake notebooks/my_notebook.ipynb

For more about ``NB_FAST`` read the :ref:`notebooks documentation
<contributing-docs-notebooks>` in :doc:`docs`.

Code Coverage
-------------

Code coverage measures how many statements of code is executed while running
tests. It identifies unused and untested code. We encourage contributors to
use it to write more robust programs, but don't have a target percentage.

To generate code coverage reports add ``--cov=capymoa`` and
``--cov-report=html`` to the pytest command:

.. code-block:: bash

    pytest --cov=capymoa --cov-report=html

Alternatively, CapyMOA's invoke testing tasks can generate coverage reports
with:

.. code-block:: bash

    invoke test --coverage

See also:

* `coverage.py <https://github.com/coveragepy/coveragepy>`_: Tool for
  measuring python code coverage.
* `pytest-cov <https://pypi.org/project/pytest-cov/>`_: PyTest plugin to
  automatically collect code coverage information with coverage.py.
