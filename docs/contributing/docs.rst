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

Versioned Releases
-------------------

Every tagged release of CapyMOA publishes its documentation at
``capymoa.org/vX.Y.Z/``. The root of the site, ``capymoa.org/``, always mirrors the
most recently released version. Older versions stay available through the version
switcher in the top navigation bar.

The ``Release`` GitHub Actions workflow (``.github/workflows/release.yml``) handles
this. Contributors do not need to do anything to trigger it:

#. The ``documentation`` job builds the docs with ``invoke docs.build``.
#. The ``publish_docs_asset`` job packages that build into ``docs-vX.Y.Z.tar.gz`` and
   attaches it to the GitHub Release for that tag, alongside the PyPI distribution
   files.
#. The ``website`` job runs three subcommands of ``docs/release_scripts.py``:

   * ``list-versions`` lists every GitHub Release that has a docs asset, then applies
     the retention policy below.
   * ``assemble`` downloads each kept release's docs asset and extracts it into its
     own ``/vX.Y.Z/`` folder, copying the newest into the site root.
   * ``build-switcher`` writes ``switcher.json`` for the version switcher.

   The job then deploys the result.

GitHub Releases are the only place a version's docs are stored. There is no
``gh-pages`` branch, so ``git clone`` of the ``capymoa`` repository stays unaffected
regardless of how many releases exist.

Docs are **never rebuilt** once published for a given version. If you spot an error
in a past version's docs, it is only fixed in the next release's docs, not
retroactively.

Docs published to Pull Request previews (see "Pull Request Artifact" above) are
**not** versioned and are **not** part of ``capymoa.org``. Versioning only applies to
tagged releases handled by the ``Release`` workflow.

Retention
~~~~~~~~~

Not every release stays live forever. ``list-versions`` groups releases by major
version (the ``X`` in ``vX.Y.Z``) and keeps:

* The last 5 releases of the current major version.
* Only the latest release of every earlier major version.

An older release that ages out of this window stops being part of the live site.
Its docs archive (``docs-vX.Y.Z.tar.gz``) stays downloadable forever from that
release's GitHub Releases page. Nothing is deleted, it is just no longer rebuilt.
Visiting a pruned version's URL on ``capymoa.org`` shows a themed "page not found"
page (``docs/404.rst``) with a link to the latest docs and to GitHub Releases for the
archived download.

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
      - | ``:mod:`capymoa.stream```
        | :mod:`capymoa.stream`
    * - Class
      - | ``:class:`capymoa.stream.Stream```
        | :class:`capymoa.stream.Stream`
    * - Method
      - | ``:meth:`capymoa.stream.Stream.next_instance```
        | :meth:`capymoa.stream.Stream.next_instance`
    * - Function
      - | ``:func:`capymoa.stream.stream_from_file```
        | :func:`capymoa.stream.stream_from_file`
    * - Attribute
      - | ``:attr:`capymoa.stream.Schema.dataset_name```
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

You can include LaTeX mathematical equations directly with the `math directive <https://docutils.sourceforge.io/docs/ref/rst/directives.html#math>`_.

..  code-block:: rst

    Block equation:

    ..  math::

        E = mc^2

    Inline equation: :math:`E = mc^2`.

Block equation:

..  math::

    E = mc^2

Inline equation: :math:`E = mc^2`.

.. _contributing-docs-notebooks:

Notebooks
---------

CapyMOA documentation includes notebooks for tutorials and narrative
documentation. Notebooks are stored as `Jupytext <https://jupytext.org>`_
``py:percent`` scripts (``notebooks/*/*.py``) (not ``.ipynb`` files). Jupytext
reduces diff sizes, works better with agents, and keeps the CapyMOA repository
smaller. (With ``.ipynb`` files, notebooks make up 90% of the repository size.)

Execute and build documentation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Executing notebooks and building the Sphinx HTML are two separate steps,
to separate notebook and Sphinx errors:

1.  ``invoke docs.nb [--slow]``: executes `.py` notebooks and generates ``.ipynb``
    with populated output cells. When ran with ``--slow`` the execution is run without
    the ``NB_FAST`` environment variable enabling mock data. GitHub pull request actions
    are tested in fast mode. GitHub releases are ran with ``--slow`` to generate the
    full outputs.

2.  ``invoke docs.build``: compiles the documentation including all
    executed notebooks.

Jupytext
~~~~~~~~

Many IDEs, including VSCode and Jupyter Lab, support Jupytext's
``py:percent`` format:

.. code-block:: python

    # %% [markdown]

For example, to open a ``.py`` notebook directly in Jupyter, install the `Jupytext
Jupyter extension <https://jupytext.readthedocs.io/en/latest/install.html>`_.
The notebook then works like a classic ``.ipynb`` file.
However, you might find it less convenient than a standard notebook.
You can convert between the two formats as needed:

..  code-block:: sh

    # Convert ipynb to py:percent
    jupytext --to py:percent notebook.ipynb

    # Convert py:percent to ipynb
    jupytext --to notebook notebook.py

    # Sync both formats (keep both files updated together)
    jupytext --set-formats ipynb,py:percent notebook.ipynb
    jupytext --sync notebook.ipynb

If you used ``uv`` for setup, prefix these commands with ``uv run jupytext
...`` or ``uv run --with docs jupytext``.

**Don't commit classic ``.ipynb`` notebooks to the repository.**

To add a notebook to the documentation:

1.  Add the ``.py`` script to the ``/notebooks`` directory, in the
    appropriate domain subdirectory (for example, ``notebooks/classifier/``).
    Add the notebook name to the ordered ``toctree`` in that directory's
    ``index.md``. The index controls the tutorial order.

2.  Write Markdown cells using `MyST Markdown
    <https://myst-parser.readthedocs.io/>`_ syntax. This lets you use Sphinx
    features such as cross-referencing, which weren't available with
    ``.ipynb`` files:

    * ``{doc}issuing/guide``
    * ``[installation guide]({doc}`install`)``

3.  To check the notebook runs without error, or to generate a matching
    ``.ipynb`` file to open, run ``invoke docs.nb``. This regenerates an
    ``.ipynb`` file alongside each ``.py`` file and executes it, writing the
    outputs back into that ``.ipynb`` file.

4.  Build the documentation locally to confirm your notebook converts and
    displays correctly. See :doc:`/contributing/docs`.


Slow Notebooks
~~~~~~~~~~~~~~

Some notebooks may take a long time to run. Here's how we handle slow notebooks:

* The ``NB_FAST`` environment variable is set to ``true`` when the notebooks
  should be run quickly. ``invoke docs.nb`` sets it for you by default
  (pass ``--slow`` to run against full-size datasets instead).

* Add hidden cells that check ``NB_FAST`` and speed up the notebook by using
  smaller datasets or fewer iterations.

* For example, you can add the following cell to the top of a notebook to replace
  some large datasets with smaller ones. You should ensure the cell is hidden on
  the website (See :ref:`hide-cells`).

    ..  code-block:: python

        # %% tags=["remove-cell"]
        # This cell is hidden on capymoa.org. See docs/contributing/docs.rst
        from capymoa._nbmock import mock_datasets, is_nb_fast
        if is_nb_fast():
            mock_datasets()

  ``capymoa._nbmock`` (``src/capymoa/_nbmock.py``) ships inside the
  ``capymoa`` package itself rather than living alongside the notebooks, so it
  is importable regardless of how a notebook is run: as a script, through
  Jupyter, or via nbmake.

.. _hide-cells:

Hide Cells
~~~~~~~~~~

You can remove a cell from being rendered on the website by tagging it
``remove-cell``, following the `MyST-NB cell tag conventions
<https://myst-nb.readthedocs.io/en/latest/render/hiding.html>`_:

..  code-block:: python

    # %% tags=["remove-cell"]
    ...

Testing Notebooks
~~~~~~~~~~~~~~~~~

The ``tasks.py`` defines a task for running the notebooks as tests:

.. code-block:: bash

    invoke docs.nb # add --help for options

.. program-output:: python -m invoke docs.nb --help

Running notebooks (via ``docs.nb``) leaves generated ``.ipynb`` files and
other side-effect files (plots, logs, TensorBoard ``runs/`` directories,
etc.) scattered under ``notebooks/*/``. Clean these up with:

.. code-block:: bash

    invoke clean.nb

Manual Documentation
--------------------

Manually written documentation is put in the ``/docs`` directory. These can be written in
reStructuredText or Markdown. To add a new page to the documentation, add a new
file to the ``/docs`` directory and add the filename to the ``toctree`` in ``index.rst``
or the appropriate location in the documentation.
