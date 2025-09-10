.. CapyMOA documentation master file, created by
   sphinx-quickstart on Fri Feb 23 08:41:28 2024.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.


CapyMOA
=======

.. image:: /images/CapyMOA.jpeg
   :alt: CapyMOA

.. image:: https://img.shields.io/pypi/v/capymoa
   :target: https://pypi.org/project/capymoa/
   :alt: Link to PyPI
   
.. image:: https://img.shields.io/discord/1235780483845984367?label=Discord
   :target: https://discord.gg/spd2gQJGAb
   :alt: Link to Discord

.. image:: https://img.shields.io/github/stars/adaptive-machine-learning/CapyMOA?style=flat
   :target: https://github.com/adaptive-machine-learning/CapyMOA
   :alt: Link to GitHub

Machine learning library tailored for data streams. Featuring a Python API
tightly integrated with MOA (**Stream Learners**), PyTorch (**Neural
Networks**), and scikit-learn (**Machine Learning**). CapyMOA provides a
**fast** python interface to leverage the state-of-the-art algorithms in the
field of data streams.

To setup CapyMOA, simply install it via pip. If you have any issues with the
installation (like not having Java installed) or if you want GPU support, please
refer to the :ref:`installation`. Once installed take a look at the
:ref:`tutorials` to get started.


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

.. image:: /images/arf100_cpu_time.png
   :alt: Performance plot
   :align: center
   :class: only-light

.. image:: /images/arf100_cpu_time_dark.png
   :alt: Performance plot
   :align: center
   :class: only-dark

Benchmark comparing CapyMOA against other data stream libraries. The benchmark
was performed using an ensemble of 100 ARF learners trained on
:class:`capymoa.datasets.RTG_2abrupt` dataset containing 100,000 samples and 30
features.  You can find the code to reproduce this benchmark in
`benchmarking.py <https://github.com/adaptive-machine-learning/CapyMOA/blob/main/notebooks/benchmarking.py>`_.
*CapyMOA has the speed of MOA with the flexibility of Python and the richness of
Python's data science ecosystem.*

📖 Cite Us
--------------

If you use CapyMOA in your research, please cite us using the following Bibtex entry::

   @misc{
      gomes2025capymoaefficientmachinelearning,
      title={{CapyMOA}: Efficient Machine Learning for Data Streams in Python},
      author={Heitor Murilo Gomes and Anton Lee and Nuwan Gunasekara and Yibin Sun and Guilherme Weigert Cassales and Justin Jia Liu and Marco Heyden and Vitor Cerqueira and Maroua Bahri and Yun Sing Koh and Bernhard Pfahringer and Albert Bifet},
      year={2025},
      eprint={2502.07432},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2502.07432}
   }

.. _installation:

🚀 Installation
---------------

Installation instructions for CapyMOA:

.. toctree::
   :maxdepth: 2

   installation
   docker

🎓 Tutorials
------------
Tutorials to help you get started with CapyMOA.

.. toctree::
   :maxdepth: 2

   tutorials

📚 Reference Manual
-------------------
Reference documentation describing the interfaces fo specific classes, functions,
and modules.

.. toctree::
   :maxdepth: 2

   api/index

ℹ️ About us
-----------

.. toctree::
   about

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
