.. toctree::
   :hidden:

   setup/index
   tutorials
   api/index
   about
   contributing/index

CapyMOA
=======

.. image:: /images/CapyMOA.jpeg
   :alt: CapyMOA

.. image:: https://img.shields.io/pypi/v/capymoa
   :target: https://pypi.org/project/capymoa/
   :alt: Link to PyPI

.. image:: https://coveralls.io/repos/github/adaptive-machine-learning/CapyMOA/badge.svg?branch=main
   :target: https://coveralls.io/github/adaptive-machine-learning/CapyMOA?branch=main
   
.. image:: https://img.shields.io/discord/1235780483845984367?label=Discord
   :target: https://discord.gg/spd2gQJGAb
   :alt: Link to Discord

.. image:: https://img.shields.io/github/stars/adaptive-machine-learning/CapyMOA?style=flat
   :target: https://github.com/adaptive-machine-learning/CapyMOA
   :alt: Link to GitHub

.. image:: https://img.shields.io/docker/v/tachyonic/jupyter-capymoa/latest?logo=docker&label=Docker&color=blue
   :target: https://hub.docker.com/r/tachyonic/jupyter-capymoa
   :alt: Docker Image Version (tag)

**CapyMOA does efficient machine learning for data streams in Python.** A data stream is
a sequences of items ariving one-by-one that is too large to efficiently process
non-sequentially. CapyMOA is a toolbox of methods and evaluators for: classification,
regression, clustering, anomaly detection, semi-supervised learning, online continual
learning, and drift detection for data streams.

Install with pip:

.. code-block:: bash

   pip install capymoa

Refer to the :ref:`setup` guide for other options, including CPU-only and dev dependencies.

.. code-block:: python

   from capymoa.datasets import Electricity
   from capymoa.classifier import HoeffdingTree
   from capymoa.evaluation import prequential_evaluation

   # 1. Load a streaming dataset
   stream = Electricity()

   # 2. Create a machine learning model
   model = HoeffdingTree(stream.get_schema())

   # 3. Run with test-then-train evaluation
   results = prequential_evaluation(stream, model)

   # 3. Success!
   print(f"Accuracy: {results.accuracy():.2f}%")

Next, we recomend the :ref:`tutorials`.

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

.. figure:: /images/arf100_cpu_time.png
   :alt: Performance plot
   :align: center
   :class: only-light

   Benchmark comparing CapyMOA against other data stream libraries [#f1]_.

.. figure:: /images/arf100_cpu_time_dark.png
   :alt: Performance plot
   :align: center
   :class: only-dark

   Benchmark comparing CapyMOA against other data stream libraries [#f1]_.

.. warning::

   CapyMOA is still in the early stages of development. The API is subject to
   change until version 1.0.0. If you encounter any issues, please report them
   on the `GitHub Issues <https://github.com/adaptive-machine-learning/CapyMOA/issues>`_
   page or talk to us on `Discord <https://discord.gg/spd2gQJGAb>`_.

.. [#f1]
   Benchmark comparing CapyMOA against other data stream libraries. The benchmark was
   performed using an ensemble of 100 ARF learners trained on
   :class:`capymoa.datasets.RTG_2abrupt` dataset containing 100,000 samples and 30
   features.  You can find the code to reproduce this benchmark in 
   `benchmarking.py <https://github.com/adaptive-machine-learning/CapyMOA/blob/main/notebooks/benchmarking.py>`_.
