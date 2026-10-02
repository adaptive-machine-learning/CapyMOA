"""Ready-made inputs for drift detectors.

A drift detector consumes one number per instance. Deciding which number is the
interesting part: the prediction error of a model, a single input feature, or
anything else you can compute. :class:`capymoa.stream.preprocessing.pipeline.DriftDetectorPipelineElement`
takes a callable for this, and these are the common ones so that you declare
intent instead of writing the callable yourself.

Each is a factory: call it to get the callable.

>>> from capymoa.classifier import OnlineBagging
>>> from capymoa.datasets import ElectricityTiny
>>> from capymoa.drift.detectors import ADWIN
>>> from capymoa.drift.monitors import prediction_is_correct
>>> from capymoa.stream.preprocessing import ClassifierPipeline
>>> stream = ElectricityTiny()
>>> pipeline = (
...     ClassifierPipeline()
...     .add_classifier(OnlineBagging(schema=stream.get_schema(), ensemble_size=5))
...     .add_drift_detector(ADWIN(), prediction_is_correct())
... )

Pass your own callable to ``add_drift_detector`` for anything these do not
cover. The signature is ``(instance, prediction)``.
"""

from __future__ import annotations

import warnings
from collections.abc import Callable
from typing import Any

from capymoa.core import (
    Instance,
    LabeledInstance,
    LabelIndex,
    RegressionInstance,
    TargetValue,
)

__all__ = ["absolute_error", "feature_value", "prediction_is_correct"]

#: Consecutive ``None`` predictions tolerated before warning. A learner returns
#: ``None`` until it can predict, which is a handful of instances; a monitor
#: placed where no prediction reaches it gets ``None`` forever.
_DEFAULT_WARN_AFTER = 100

_MISPLACED_HINT = (
    "received {count} consecutive predictions of None. A learner returns None "
    "only until it can predict, so this usually means the drift detector sits "
    "before the learner in the pipeline and no prediction reaches it. Move the "
    "detector after the learner, or monitor an input feature with "
    "capymoa.drift.monitors.feature_value instead."
)


def _none_watcher(name: str, warn_after: int) -> Callable[[bool], None]:
    """Return a function that warns once after `warn_after` consecutive Nones.

    Counting consecutive rather than total ``None`` is what separates a learner
    warming up from a detector that will never see a prediction.
    """
    state = {"consecutive": 0, "warned": False}

    def observe(prediction_is_none: bool) -> None:
        if not prediction_is_none:
            state["consecutive"] = 0
            return
        state["consecutive"] += 1
        if not state["warned"] and state["consecutive"] >= warn_after:
            state["warned"] = True
            warnings.warn(
                f"{name}: " + _MISPLACED_HINT.format(count=state["consecutive"]),
                UserWarning,
                stacklevel=3,
            )

    return observe


def prediction_is_correct(
    warn_after: int = _DEFAULT_WARN_AFTER,
) -> Callable[[LabeledInstance, LabelIndex | None], int]:
    """Monitor whether a classifier predicted the label correctly.

    Returns ``1`` for a correct prediction and ``0`` otherwise, which is what a
    detector such as :class:`capymoa.drift.detectors.ADWIN` expects when
    monitoring accuracy.

    A ``None`` prediction counts as incorrect. Classifiers return ``None`` until
    they can predict, so the first few instances of a stream are ``None`` even
    when everything is wired correctly. After ``warn_after`` *consecutive*
    ``None`` predictions a :class:`UserWarning` is raised once, because that
    means no prediction is reaching the monitor at all.

    :param warn_after: Consecutive ``None`` predictions tolerated before warning.
    :return: A callable taking ``(instance, prediction)``.

    >>> from capymoa.datasets import ElectricityTiny
    >>> from capymoa.drift.monitors import prediction_is_correct
    >>> instance = ElectricityTiny().next_instance()
    >>> monitor = prediction_is_correct()
    >>> monitor(instance, instance.y_index)
    1
    >>> monitor(instance, 1 - instance.y_index)
    0
    """
    observe = _none_watcher("prediction_is_correct", warn_after)

    def monitor(instance: LabeledInstance, prediction: LabelIndex | None) -> int:
        observe(prediction is None)
        return int(prediction is not None and instance.y_index == prediction)

    return monitor


def absolute_error(
    warn_after: int = _DEFAULT_WARN_AFTER,
) -> Callable[[RegressionInstance, TargetValue | None], float]:
    """Monitor the absolute prediction error of a regressor.

    The regression counterpart of :func:`prediction_is_correct`.

    A ``None`` prediction contributes ``0.0``. Note the direction of that
    choice: zero error looks like a perfect prediction, so it biases the
    detector *away* from reporting drift. The warning described in
    :func:`prediction_is_correct` is what guards against a monitor that only
    ever sees ``None``.

    :param warn_after: Consecutive ``None`` predictions tolerated before warning.
    :return: A callable taking ``(instance, prediction)``.

    >>> from capymoa.datasets import FriedTiny
    >>> from capymoa.drift.monitors import absolute_error
    >>> instance = FriedTiny().next_instance()
    >>> monitor = absolute_error()
    >>> monitor(instance, instance.y_value)
    0.0
    >>> monitor(instance, instance.y_value + 2.5)
    2.5
    """
    observe = _none_watcher("absolute_error", warn_after)

    def monitor(instance: RegressionInstance, prediction: TargetValue | None) -> float:
        observe(prediction is None)
        if prediction is None:
            return 0.0
        return float(abs(instance.y_value - prediction))

    return monitor


def feature_value(index: int) -> Callable[[Instance, Any], float]:
    """Monitor one input feature, ignoring the prediction.

    Use this for unsupervised drift detection: the detector watches the data
    itself rather than a model's performance, so it works wherever you place it
    and does not need a learner upstream.

    The feature is read *as it reaches this point in the pipeline*, so the same
    index gives different values before and after a transformer.

    :param index: Position in :attr:`capymoa.core.Instance.x`.
    :return: A callable taking ``(instance, prediction)``; the prediction is
        ignored.

    >>> from capymoa.datasets import ElectricityTiny
    >>> from capymoa.drift.monitors import feature_value
    >>> instance = ElectricityTiny().next_instance()
    >>> monitor = feature_value(0)
    >>> monitor(instance, None) == float(instance.x[0])
    True
    """

    def monitor(instance: Instance, prediction: Any = None) -> float:
        return float(instance.x[index])

    return monitor
