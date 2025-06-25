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

Pull Request Artifact
---------------------

Reviewers and developers can preview the documentation of a pull request by
downloading the documentation artifact, extracting, and then opening it in a
browser. You can download the documentation artifact from "Pull Request"
workflow.

..  seealso::

    `Downloading Workflow Artifact <https://docs.github.com/en/actions/managing-workflow-runs-and-deployments/managing-workflow-runs/downloading-workflow-artifacts>`_

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

Here is an example of how to write a docstring for a classifier in CapyMOA:

.. code-block:: python

    from capymoa.base import Classifier
    from capymoa.stream import Schema


    class ExampleClassifier(Classifier):
        """One line docstring.

        You may add a multi-line detailed description of the classifier. You
        should include a citation [#example25]_ to the source paper.

        You may include an example of how to use the classifier. This example is
        serves as both documentation and a test for the classifier. Keep in mind
        that these are run as part of the test suite, so they should be kept
        simple, deterministic, and fast.

        >>> from capymoa.datasets import ElectricityTiny
        >>> from capymoa.classifier import ExampleClassifier
        >>> from capymoa.evaluation import prequential_evaluation
        >>> stream = ElectricityTiny()
        >>> learner = ExampleClassifier(stream.get_schema())
        >>> results = prequential_evaluation(stream, learner, max_instances=1000)
        >>> results["cumulative"].accuracy()
        87.9

        You may include a see also section with links to related classes or
        functions. This is useful for users to find related functionality in the
        library.

        .. seealso::

            :func:`capymoa.evaluation.prequential_evaluation`

        .. [#example25] Example, A., Author, B., & Researcher, C. (2025). Example Classifier.
        """

        class_attr = None
        """One-line docstring for ``class_attr``."""

        def __init__(self, schema: Schema):
            """Construct a new ExampleClassifier.

            :param schema: Describes the structure of the data stream.
            """
            super().__init__(schema)

            #: One-line docstring for ``attr_a``.
            self.attr_a = None

            self.attr_b = None
            """Another syntax for a one-line docstring."""

            self.attr_c = None
            """Multi-line docstring for ``attr_c`` attribute.

            It can include multiple lines and is useful for providing detailed
            information about the attribute's purpose and usage.
            """

For exemplars take a look at the docstrings in the
:class:`~capymoa.classifier.AdaptiveRandomForestClassifier` or 
:class:`~capymoa.classifier.HoeffdingAdaptiveTree` classes.

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



Citations
~~~~~~~~~

You should reference sources using the `reStructuredText footnotes syntax
<https://docutils.sourceforge.io/docs/ref/rst/restructuredtext.html#footnotes>`_.
We prefer footnotes over citations since they are local to the page and do not
require a global bibliography. This makes it easier to read the documentation
without having to jump between pages.


For example, to reference a source in the text:

.. code-block:: rst

    CapyMOA is a Python library for efficient machine learning on data
    streams [#gomes25]_.

    .. [#gomes25] Gomes, H. M., Lee, A., Gunasekara, N., Sun, Y., Cassales, G. W.,
        Liu, J., Heyden, M., Cerqueira, V., Bahri, M., Koh, Y. S., Pfahringer,
        B., & Bifet, A. (2025). CapyMOA: Efficient machine learning for data
        streams in python. CoRR, abs/2502.07432.
        https://doi.org/10.48550/ARXIV.2502.07432

CapyMOA is a Python library for efficient machine learning on data
streams [#gomes25]_.

.. [#gomes25] Gomes, H. M., Lee, A., Gunasekara, N., Sun, Y., Cassales, G. W.,
    Liu, J., Heyden, M., Cerqueira, V., Bahri, M., Koh, Y. S., Pfahringer,
    B., & Bifet, A. (2025). CapyMOA: Efficient machine learning for data
    streams in python. CoRR, abs/2502.07432.
    https://doi.org/10.48550/ARXIV.2502.07432


Cross Reference
~~~~~~~~~~~~~~~

You can link to the documentation of a module, class, method, function, 
attribute, or other programming constructs using the `sphinx cross-reference syntax <https://www.sphinx-doc.org/en/master/usage/referencing.html>`_.

..  list-table::
    :widths: 20 80

    * - Module
      - | ``:mod:`capymoa.stream``
        | :mod:`capymoa.stream`
    * - Class
      - | ``:class:`capymoa.stream.Stream``
        | :class:`capymoa.stream.Stream`
    * - Method
      - | ``:meth:`capymoa.stream.Stream.next_instance``
        | :meth:`capymoa.stream.Stream.next_instance`
    * - Function
      - | ``:func:`capymoa.stream.stream_from_file```
        | :func:`capymoa.stream.stream_from_file`
    * - Attribute
      - | ``:attr:`capymoa.stream.Schema.dataset_name``
        | :attr:`capymoa.stream.Schema.dataset_name`

Add the prefix ``~`` to the name to display the name without the prefixing path:

..  code-block:: rst

    :meth:`~capymoa.stream.Stream.next_instance`

This will display as :meth:`~capymoa.stream.Stream.next_instance`.


..  seealso::

    `Sphinx Cross-referencing <https://www.sphinx-doc.org/en/master/usage/referencing.html>`_

    `Sphinx Cross-referencing Python Objects <https://www.sphinx-doc.org/en/master/usage/domains/python.html#cross-referencing-python-objects>`_


See Also
~~~~~~~~

It can be handy to link to related documentation pages or external resources without
explicitly referencing them in the text. This can be done using the 
`sphinx seealso directive <https://www.sphinx-doc.org/en/master/usage/restructuredtext/directives.html#directive-seealso>`_.

..  code-block:: rst

    ..  seealso::

        `See Also <https://www.sphinx-doc.org/en/master/usage/restructuredtext/directives.html#directive-seealso>`_
            Documents Sphinx seealso directive.

        `Definition List  <https://docutils.sourceforge.io/docs/ref/rst/restructuredtext.html#definition-lists>`_
            Documents reStructuredText definition lists.

..  seealso::

    `See Also <https://www.sphinx-doc.org/en/master/usage/restructuredtext/directives.html#directive-seealso>`_
        Documents Sphinx seealso directive.

    `Definition List  <https://docutils.sourceforge.io/docs/ref/rst/restructuredtext.html#definition-lists>`_
        Documents reStructuredText definition lists.

Math
~~~~

You can include LaTex mathematical equations directly with the `math directive <https://docutils.sourceforge.io/docs/ref/rst/directives.html#math>`_.

..  code-block:: rst

    Block equation:

    ..  math::

        E = mc^2

    Inline equation: :math:`E = mc^2`.

Block equation:

..  math::

    E = mc^2

Inline equation: :math:`E = mc^2`.

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
