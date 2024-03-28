# Adding Tests
Ensure you have installed the development dependencies by following the instructions
in the [installation guide](installation.md). To run all tests, use the following command:
```bash
invoke test
```

## PyTest
Tests can be added to the ``tests`` directory. PyTest will automatically discover
and run these tests. They should be named ``test_*.py`` and the test functions
should be named ``test_*``. See the [PyTest documentation](https://docs.pytest.org) 
for more information.

Use this PyTest style tests for parameterized tests, tests that require fixtures,
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
invoke test.unit
```

## Doctest
[Doctest](https://docs.python.org/3/library/doctest.html) allows you to write 
tests directly in the docstrings of your code, making it easier to keep documentation
up-to-date. The tests are written in the form of examples in a Python interactive shell.

Use doctest style test to document code with simple tested examples.

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
pytest --doctest-modules -k src
```

The `-k src` flag tells pytest to only run tests in the `src` directory. This is useful
if you only want to run doctests in source code but not PyTest tests.

Alternatively, you can run all unit tests with the same configuration as continuous integration:
```bash
invoke test.unit
```


## Notebooks
Tests can be added to the `notebooks` directory. These tests are written in the form of
[Jupyter Notebooks](https://jupyter.org) and are run using the `nbmake`
extension for PyTest.

Use this type of test for tests that provide tutorials, documentation, and
examples. They should be easy to read and understand.

These test can be run just these tests with:

```bash
invoke test.nb
```

If you need to overwrite the notebooks with the output, use:

```bash
invoke test.nb --overwrite
```



