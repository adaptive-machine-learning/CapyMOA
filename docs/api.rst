.. _api:
API
===

.. module:: capymoa

This part of the documentation describes the interfaces for CapyMOA classes,
functions, and modules. If you are looking to just use CapyMOA, you should start
with the :ref:`tutorials<tutorials>`.


Instance
--------
Instances are the basic unit of data in CapyMOA.

.. autoclass:: capymoa.stream.instance.Instance
    :members:
    :inherited-members:

.. autoclass:: capymoa.stream.instance.LabeledInstance
    :members:
    :inherited-members:
    :show-inheritance:

.. autoclass:: capymoa.stream.instance.RegressionInstance
    :members:
    :inherited-members:
    :show-inheritance:

Stream
------
A datastream is a sequence of instances arriving one at a time.


.. automodule:: capymoa.stream
    :members:
    :undoc-members:
    :inherited-members:



Learners
--------
CapyMOA defines different interfaces for learners performing different machine
learning tasks.

.. autoclass:: capymoa.learner.learners.Classifier
    :members:
    :undoc-members:
    :inherited-members:

.. autoclass:: capymoa.learner.learners.Regressor
    :members:
    :undoc-members:
    :inherited-members:

.. autoclass:: capymoa.learner.learners.ClassifierSSL
    :members:
    :undoc-members:
    :inherited-members:

Classifiers
-----------
Classifiers implement the :class:`capymoa.learner.learners.Classifier` interface.

.. automodule:: capymoa.learner.classifier
    :members:
    :undoc-members:
    :show-inheritance:
    :inherited-members:

Semi-Supervised Classifiers
---------------------------
Semi-Supervised classifiers implement the :class:`capymoa.learner.learners.ClassifierSSL` interface.

.. automodule:: capymoa.learner.ssl.classifier
    :members:
    :undoc-members:
    :show-inheritance:
    :inherited-members:

Regressors
----------
Regressors implement the :class:`capymoa.learner.learners.Regressor` interface.

.. automodule:: capymoa.learner.regressor
    :members:
    :undoc-members:
    :show-inheritance:
    :inherited-members:

Evaluation
----------
CapyMOA provides comes with some evaluation methods to evaluate the performance of
learners.

.. automodule:: capymoa.evaluation.evaluation
    :members:
    :undoc-members:
    :show-inheritance:
    :inherited-members:

Datasets
--------
CapyMOA comes with some datasets 'out of the box'. These are easily imported
and used being downloaded the first time you use them.

.. automodule:: capymoa.datasets
    :members:
    :undoc-members:
    :show-inheritance:
    :inherited-members: