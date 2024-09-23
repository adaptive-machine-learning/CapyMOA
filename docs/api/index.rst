API Reference
=============

Welcome to the capymoa API reference. This documentation is automatically
generated from the source code and provides detailed information on the classes
and functions available in capymoa. 

If you are looking to just use CapyMOA, you should start with the
:ref:`tutorials<tutorials>`.

Types
-----

These module provide interfaces for learners, and other basic types used by
capymoa.

..  autosummary::
    :toctree: modules
    :caption: Types
    :recursive:

    capymoa.base
    capymoa.type_alias
    capymoa.instance

Data Streams
------------

These modules provide classes for loading, and simulating data streams. It also
includes utilities for simulating concept drifts.

..  autosummary::
    :toctree: modules
    :caption: Data Streams
    :recursive:

    capymoa.datasets
    capymoa.stream

Learners
--------

These modules implement learners for classification, regression, anomaly detection
and semi-supervised learning.
    
..  autosummary::
    :toctree: modules
    :caption: Learners
    :recursive:

    capymoa.classifier
    capymoa.regressor
    capymoa.anomaly
    capymoa.ssl

Drift Detection
---------------

These modules provide classes for detecting concept drifts.

..  autosummary::
    :toctree: modules
    :caption: Drift Detection
    :recursive:

    capymoa.drift

Evaluation
----------

These modules provide classes for evaluating learners.

..  autosummary::
    :toctree: modules
    :caption: Evaluation
    :recursive:

    capymoa.splitcriteria
    capymoa.evaluation
    capymoa.prediction_interval

Miscellaneous
-------------

These modules provide miscellaneous utilities.

..  autosummary::
    :toctree: modules
    :caption: Miscellaneous
    :recursive:

    capymoa.misc
    capymoa.env

Functions
---------

..  automodule:: capymoa
    :members: