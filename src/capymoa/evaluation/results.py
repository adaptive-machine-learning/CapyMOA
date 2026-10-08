"""Shared plumbing for results.

Every research domain defines its own flat, typed result (see
:class:`capymoa.classifier.ClassifierResults`). All of them extend
:class:`RunInfo`. A result is a plain :class:`dict`, so it can be pickled,
compared and put in a :class:`pandas.DataFrame`.

>>> from capymoa.classifier import HoeffdingTree, NaiveBayes, evaluate_classifiers
>>> from capymoa.datasets import ElectricityTiny
>>> import pandas as pd
>>> stream = ElectricityTiny()
>>> results = evaluate_classifiers(
...     stream,
...     {"ht": HoeffdingTree(stream.get_schema()), "nb": NaiveBayes(stream.get_schema())},
...     max_instances=1000,
... )
>>> pd.DataFrame(list(results.values()), index=list(results))[["accuracy"]].round(1)
    accuracy
ht      84.4
nb      84.8
"""

from typing import NotRequired, TypedDict

import numpy as np


class Windows(TypedDict):
    """The windowed metrics of a run, in columns.

    A dict of equal length arrays: one entry per window. Make a table with
    ``pd.DataFrame(windows)`` and go back with
    ``{c: df[c].to_numpy() for c in df}``. Each domain adds one key for each of
    its metrics.
    """

    #: The number of instances seen at the end of each window.
    instances: np.ndarray


class ConceptSpan(TypedDict):
    """Where a concept of a :class:`~capymoa.stream.drift.RecurrentConceptDriftStream` lasts."""

    #: Name of the concept. A concept that recurs has the same name each time.
    id: str
    #: Instance index where the concept starts.
    start: int
    #: Instance index where the concept ends.
    end: int


class RunInfo(TypedDict):
    """Information about a run, shared by the results of every domain."""

    #: Name of the learner. The key when many learners were evaluated together.
    learner: str
    #: Name of the stream (not the stream object, so results can be serialised).
    stream: str
    #: Number of instances the learner was evaluated on.
    instances: int
    #: Number of instances in a window. Absent if windowed results are off.
    window_size: NotRequired[int]
    #: Elapsed time in seconds.
    wallclock: float
    #: CPU time in seconds.
    cpu_time: float
    #: The metrics of each window, in columns (see :class:`capymoa.evaluation.results.Windows`). The result
    #: of a domain narrows this to its own metrics. Absent if ``window_size`` is
    #: absent.
    windowed: NotRequired[Windows]
    #: The ground truth targets. Absent unless ``store_y`` was set.
    y_true: NotRequired[np.ndarray]
    #: The predictions. Absent unless ``store_predictions`` was set.
    y_pred: NotRequired[np.ndarray]
    #: Instance indexes of the drifts in the stream. Absent if the stream has
    #: no drifts.
    drifts: NotRequired[list[int]]
    #: Width of each drift in ``drifts``, in instances. 0 or 1 for an abrupt
    #: drift. Absent if the stream has no drifts.
    drift_widths: NotRequired[list[int]]
    #: Where each concept lasts. Only for a
    #: :class:`~capymoa.stream.drift.RecurrentConceptDriftStream`.
    concepts: NotRequired[list[ConceptSpan]]
