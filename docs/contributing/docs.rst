Documentation
=============

To build the documentation, run the following command in the project root:

.. code-block:: bash

    python -m invoke docs.build

.. program-output:: python -m invoke docs.build --help

Once built, you can visit the documentation locally in your browser.

.. note::

    If you run into nitpicky errors, you can allow a more permissive documentation
    build with:

    .. code-block:: bash

        python -m invoke docs.build -i

    Continuous integration will still run the strict build, so make sure to fix
    any errors before making a pull request.

Docstrings
----------

CapyMOA uses Sphinx to generate documentation from function, class, and module
docstring comments. CapyMOA uses the `sphinx/reStructuredText
<https://sphinx-rtd-tutorial.readthedocs.io/en/latest/docstrings.html>`_ style of
docstrings. Rather than having type information in the docstring, we prefer to
use Python-type hints. This allows for better type checking and IDE support.

.. warning::

    Some parts of our codebase use the wrong docstring format (e.g. Google
    style, NumPy style, etc.). These are **wrong** since they are not parsed
    correctly by Sphinx and display strangely on the website. We are in the
    process of fixing these. **Please do not use these as examples for your own
    docstrings.**

Here is an example of a function docstring:

.. code-block:: python

    class Stream:
        """A datastream that can be learnt instance by instance."""

        def __init__(
            self,
            moa_stream: Optional[InstanceStream] = None,
            schema: Optional[Schema] = None,
            CLI: Optional[str] = None,
        ):
            """Construct a Stream from a MOA stream object.

            Usually, you will want to construct a Stream using the :func:`stream_from_file`
            function.

            :param moa_stream: The MOA stream object to read instances from. Is None
                if the stream is created from a numpy array.
            :param schema: The schema of the stream. If None, the schema is inferred
                from the moa_stream.
            :param CLI: Additional command line arguments to pass to the MOA stream.
            :raises ValueError: If no schema is provided and no moa_stream is provided.
            :raises ValueError: If command line arguments are provided without a moa_stream.
            """


.. important::

    If you use **autodocstring for VSCode**, set the docstring format to `sphinx-notypes` in the settings.
    (`autodocstring <https://marketplace.visualstudio.com/items?itemName=njpwerner.autodocstring>`_)

    If you use **PyCharm**, set the docstring format to `reStructuredText` in the settings.
    (`PyCharm settings <https://www.jetbrains.com/help/pycharm/settings-tools-python-integrated-tools.html>`_)

    If you use an AI tool to generate docstrings please ensure that it actually
    outputs reStructuredText style docstrings. Also go through the docstring and
    ensure it is **concise** and correct. You may have luck setting up a project
    wide prompt (`Copilot docs
    <https://docs.github.com/en/copilot/customizing-copilot/adding-repository-custom-instructions-for-github-copilot?tool=vscode>`_).


Notebooks
---------

CapyMOA documentation includes Jupyter Notebooks for tutorials, and narrative
style documentation. These notebooks are run as tests to ensure they are kept
up-to-date. This document explains how to run, render and test notebooks.

* Added to the ``/notebooks`` directory.
* Rendered to HTML and included in the documentation of the website using 
* To add a notebook to the documentation, add the notebook to the ``/notebooks``
  directory and add the filename to the ``toctree`` in ``notebooks/index.rst``.
* Please check the notebooks are being converted and included in the documentation
  by building the documentation locally. See :doc:`/contributing/docs`.
*   The parser for markdown used by Jupiter Notebooks is different from the one
    used by nbsphinx. This can lead to markdown rendering unexpectedly you might
    need to adjust the markdown in the notebooks to render correctly on the website.

    *   Bullet points should have a newline after the bullet point.
      
        ..  code-block:: markdown

            * Bullet point 1

            * Bullet point 2

Slow Notebooks
~~~~~~~~~~~~~~

Some notebooks may take a long time to run. Heres how we handle slow notebooks:

* The ``NB_FAST`` environment variable is set to ``Tue`` when the notebooks should
  be run quickly.

* Add hidden cells that check ``NB_FAST`` and speed up the notebook by using
  smaller datasets or fewer iterations.

*   For example, you can add the following cell to the top of a notebook to replace
    some large datasets with smaller ones. You should ensure the cell is hidden on
    the website (See :ref:`hide-cells`).

    ..  code-block:: python

        # This cell is hidden on capymoa.org. See docs/contributing/docs.rst
        from util.nbmock import mock_datasets, is_nb_fast
        if is_nb_fast():
            mock_datasets()

.. _hide-cells:

Hide Cells
~~~~~~~~~~


You can remove a cell from being rendered on the website by adding the following
to the cell's metadata:

..  code-block:: json

    "metadata": {
        "nbsphinx": "hidden"
    }


Testing or Overwriting Notebook Output
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The ``tasks.py`` defines aliases for running the notebooks as tests or for
overwriting the outputs of the notebooks. To run the notebooks as tests:

.. code-block:: bash

    invoke test.nb # add --help for options

.. program-output:: python -m invoke test.nb --help



Manual Documentation
--------------------

Manually written documentation in the ``/docs`` directory. These can be written in
reStructuredText or Markdown. To add a new page to the documentation, add a new
file to the ``/docs`` directory and add the filename to the ``toctree`` in ``index.rst``
or the appropriate location in the documentation.
