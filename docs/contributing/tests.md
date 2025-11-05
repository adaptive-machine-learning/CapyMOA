# Adding Tests

Ensure you have installed the development dependencies by following the instructions
in the [installation guide](../installation.rst). To run all tests, use the following command:

```bash
invoke test
```

(pytest)=
## PyTest

Tests can be added to the ``tests`` directory. PyTest will automatically discover
and run these tests. They should be named ``test_*.py``, and the test functions
should be named ``test_*``. See the [PyTest documentation](https://docs.pytest.org)
for more information.

Use PyTest style tests for parameterised tests, tests that require fixtures,
and tests that require setup.

These tests can be run with:

```bash
pytest
```

Or to run a specific test

```bash
pytest tests/test_*.py
```

Or to run with the same configuration as continuous integration:
```bash
invoke test.pytest
```

(doctest)=
## Doctest

[Doctest](https://docs.python.org/3/library/doctest.html) allows you to write
tests directly in the docstrings of your code, making it easier to keep documentation
up-to-date. The tests are written as examples in a Python interactive shell.

Use doctest style tests to document code with simple tested examples.

Here's an example of a function with a doctest:

```python
def hello_world():
    """
    >>> hello_world()
    Hello, World!
    """
    print("Hello, World!")
```

You can run this test with:

```bash
pytest --doctest-modules path/to/your/module.py
```

Alternatively, you can run all unit tests with the same configuration as continuous integration:

```bash
invoke test.doctest
```

(notebooks)=
## Notebooks
We use [nbmake](https://github.com/treebeardtech/nbmake) to test that all notebooks in
the `notebooks` directory run without error. This ensures that the notebooks are always
up-to-date and working correctly.

You can run a notebook as a test with:
```bash
pytest --nbmake notebooks/my_notebook.ipynb

# Often the examples take too long to run regularly as tests. To speed up testing some
# notebooks use the `NB_FAST` environment variable to run the notebook faster by using
# smaller datasets or fewer iterations. To run them in this mode use:
NB_FAST=true pytest --nbmake notebooks/my_notebook.ipynb
```

For more about `NB_FAST` read the [notebooks documentation](../docs.rst#notebooks).
