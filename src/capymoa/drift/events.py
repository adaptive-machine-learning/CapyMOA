"""Events emitted by drift detectors.

Every drift detector has an ``events`` attribute (a
:class:`~capymoa.events.Dispatcher`). Subscribe a function to be called when
the detector finds a drift:

>>> from capymoa.drift.detectors import ADWIN
>>> from capymoa.drift.events import DriftEvent
>>> detector = ADWIN()
>>> detector.events.subscribe(DriftEvent, lambda e: print("drift at", e.index))
>>> for value in [0.0] * 1000 + [1.0] * 1000:
...     detector.add_element(value)
drift at 1024
drift at 1056
drift at 1088

Subscribe to :class:`DriftOrWarningEvent` to receive both drifts and warnings.
A detector emits an event each time it records one in ``detection_index`` or
``warning_index``, after :meth:`~capymoa.drift.base_detector.BaseDriftDetector.add_element`
has updated its state. Its ``emits`` attribute lists the events it can emit.
"""

from dataclasses import dataclass
from typing import TYPE_CHECKING

from capymoa.events import Event

if TYPE_CHECKING:
    import capymoa.drift.base_detector


@dataclass
class DriftOrWarningEvent(Event):
    """A drift detector found a drift or a warning.

    Subscribe to this class to receive both :class:`DriftEvent` and
    :class:`WarningEvent`.
    """

    detector: "capymoa.drift.base_detector.BaseDriftDetector"
    """The detector that emitted the event."""
    index: int
    """The index the detector recorded, as stored in ``detection_index`` or
    ``warning_index``."""


class DriftEvent(DriftOrWarningEvent):
    """A drift detector detected a drift."""


class WarningEvent(DriftOrWarningEvent):
    """A drift detector is in its warning zone.

    Emitted for every element the detector sees while in the warning zone, as
    ``warning_index`` records.
    """


__all__ = ["DriftEvent", "DriftOrWarningEvent", "WarningEvent"]
