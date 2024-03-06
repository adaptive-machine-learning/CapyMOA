Contributing
============

Adding Tests
------------

There are three ways to add tests to CapyMOA. Lots of these tests also serve as
documentation and examples.

PyTest
~~~~~~
Tests can be added to the ``tests`` directory. PyTest will automatically discover
and run these tests. They should be named ``test_*.py`` and the test functions
should be named ``test_*``. See the `PyTest documentation <https://docs.pytest.org>`_ 
for more information.

Use this type of test for parameterized tests, tests that require fixtures,
and tests that require a lot of setup.

These tests can be run with::
    
        pytest

Or to run a specific test::

        pytest tests/test_*.py

Or to run with the same configuration as continuous integration::

        invoke test.unit

Doctest
~~~~~~~
Tests can be added to the docstrings of functions and classes using 
`doctest <https://docs.python.org/3/library/doctest.html>`_. These tests
are written in the form of examples in a python interactive shell.

Use this type of test for simple examples and as a way to document the code.

CapyMOA uses the `sphinx.ext.doctest <https://www.sphinx-doc.org/en/master/usage/extensions/doctest.html#module-sphinx.ext.doctest>`_
to add these tests in the documentation. Here is an example of a doctest::

    """
    ..  doctest:: python

        >>> print("Hello, World!")
        Hello, World!
    """

They can be run with the following command::

        pytest --doctest-modules

Or to run with the same configuration as continuous integration::

        invoke test.unit


Notebooks
~~~~~~~~~
Tests can be added to the ``notebooks`` directory. These tests are written in the form of
`Jupyter Notebooks <https://jupyter.org>`_ and are run using the `nbmake`
extension for PyTest.

Use this type of test for tests that provide tutorials, documentation, and
examples. They should be easy to read and understand.

These test can be run with::

    invoke test.nb

If you need to overwrite the notebooks with the output, use::

    invoke test.nb --overwrite



