"""Alias of :mod:`capymoa.events`, kept so existing imports work.

The event system started here and is now used outside OCL too. The OCL events
themselves are in :mod:`capymoa.ocl.evaluate.events`.
"""

# No __all__, so the API docs list these once, under capymoa.events.
from capymoa.events import Dispatcher, Event, Handler  # noqa: F401
