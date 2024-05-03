.. CapyMOA documentation master file, created by
   sphinx-quickstart on Fri Feb 23 08:41:28 2024.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

.. image:: /images/CapyMOA.jpeg
    :alt: CapyMOA

CapyMOA
=======
.. image:: https://img.shields.io/pypi/v/capymoa
   :target: https://pypi.org/project/capymoa/
   :alt: PyPI version badge

.. Discord: 
.. image:: https://img.shields.io/discord/1235780483845984367?label=Discord
    :target: https://discord.gg/spd2gQJGAb
    :alt: discord badge https://discord.gg/spd2gQJGAb

Machine learning library tailored for data streams. Featuring a Python API
tightly integrated with MOA (**Stream Learners**), PyTorch (**Neural
Networks**), and scikit-learn (**Machine Learning**). CapyMOA provides a
**fast** python interface to leverage the state-of-the-art algorithms in the
field of data streams.

To setup CapyMOA, simply install it with pip. If you have any issues with the
installation (like no Java) or if you want GPU support, please refer to the
:ref:`installation` section. Once installed take a look at the :ref:`tutorials`
to get started.

.. code-block:: bash

   # CapyMOA requires Java. This checks if you have it installed
   java -version

   # CapyMOA requires PyTorch. This installs the CPU version
   pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

   # Install CapyMOA and its dependencies
   pip install capymoa

   # Check that the install worked
   python -c "import capymoa; print(capymoa.__version__)"

.. warning::

   CapyMOA is still in the early stages of development. The API is subject to
   change until version 1.0.0. If you encounter any issues, please report them
   on the `GitHub Issues <https://github.com/adaptive-machine-learning/CapyMOA/issues>`_
   or talk to us on `Discord <https://discord.gg/spd2gQJGAb>`_.

.. image:: /images/benchmark_20240422_221824_performance_plot_wallclock.png
   :alt: Performance plot
   :align: center

.. _installation:

🚀 Installation
---------------

Installation instructions for CapyMOA:

.. toctree::
   :maxdepth: 2

   installation

🎓 Tutorials
------------
Tutorials to help you get started with CapyMOA.

.. toctree::
   :maxdepth: 2

   notebooks/index

📚 Reference Manual
-------------------
Reference documentation describing the interfaces fo specific classes, functions,
and modules.

.. toctree::
   :maxdepth: 2

   api/index

🏗️ Contributing
---------------
This part of the documentation is for developers and contributors.

.. toctree::
   :maxdepth: 2

   contributing/index

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
