"""Run your own code when something happens inside CapyMOA.

An object that emits events has an ``events`` attribute, which is a
:class:`Dispatcher`. Subscribe a function to an event class, and the dispatcher
calls it with an instance of that class each time the event happens.

>>> from capymoa.events import Dispatcher, Event
>>> class Ping(Event):
...     pass
>>> class LoudPing(Ping):
...     pass
>>> dispatcher = Dispatcher()
>>> dispatcher.subscribe(Ping, lambda event: print("got", type(event).__name__))
>>> dispatcher.notify(LoudPing())
got LoudPing

Subscribing to a class also receives its subclasses, as ``except`` does for
exceptions. Callbacks run in the order they were subscribed.

The events a class emits are listed in its documentation and in its ``emits``
attribute. For example, :class:`~capymoa.drift.base_detector.BaseDriftDetector`
emits the events in :mod:`capymoa.drift.events`.
"""

from abc import ABC, abstractmethod
from collections.abc import Callable, Iterable
from typing import TypeVar, cast


class Event:
    """Base class for events emitted by a :py:class:`Dispatcher`."""


_E = TypeVar("_E", bound=Event)


class Dispatcher:
    """Publish events to subscribed callbacks.

    Create a dispatcher, subscribe one or more callbacks for an event type, and
    send event instances with :py:meth:`notify`.

    >>> dispatcher = Dispatcher()
    >>> def handler(event: Event): print(f"Received event: {type(event).__name__}")
    >>> dispatcher.subscribe(Event, handler)
    >>> dispatcher.notify(Event())
    Received event: Event

    A dispatcher can list the events it sends. It then refuses a subscription
    that could never be called:

    >>> class Ping(Event):
    ...     pass
    >>> class Pong(Event):
    ...     pass
    >>> dispatcher = Dispatcher(emits=[Ping])
    >>> dispatcher.subscribe(Pong, handler)
    Traceback (most recent call last):
    ...
    ValueError: Pong is never emitted here. Emitted events: Ping.
    """

    def __init__(self, emits: Iterable[type[Event]] | None = None) -> None:
        """Create a dispatcher with no subscribers.

        :param emits: The event types this dispatcher sends. If given,
            :meth:`subscribe` raises for any other type. If ``None``, any type
            is accepted.
        """
        self.emits: tuple[type[Event], ...] | None = (
            None if emits is None else tuple(emits)
        )
        """The event types this dispatcher sends, or ``None`` if not declared."""
        self._subscriptions: list[tuple[type[Event] | None, Callable[[Event], None]]]
        self._subscriptions = []

    def subscribe(
        self, event_type: type[_E] | None, callable: Callable[[_E], None]
    ) -> None:
        """Call ``callable`` for every event of ``event_type`` or a subclass.

        :param event_type: The event class to receive, or ``None`` for all events.
        :param callable: Called with the event as its only argument.
        :raises TypeError: If ``event_type`` is not an :class:`Event` subclass.
        :raises ValueError: If :attr:`emits` is set and no emitted type is
            ``event_type`` or a subclass of it.
        """
        if event_type is not None:
            if not (isinstance(event_type, type) and issubclass(event_type, Event)):
                raise TypeError(
                    f"event_type must be an Event subclass or None, got {event_type!r}"
                )
            if self.emits is not None and not any(
                issubclass(emitted, event_type) for emitted in self.emits
            ):
                names = ", ".join(emitted.__name__ for emitted in self.emits)
                raise ValueError(
                    f"{event_type.__name__} is never emitted here. "
                    f"Emitted events: {names}."
                )
        callable = cast(Callable[[Event], None], callable)
        self._subscriptions.append((event_type, callable))

    def unsubscribe(
        self, event_type: type[_E] | None, callable: Callable[[_E], None]
    ) -> None:
        """Remove a callback added with :meth:`subscribe`.

        :param event_type: The event class it was subscribed to.
        :param callable: The callback to remove.
        :raises ValueError: If ``callable`` is not subscribed to ``event_type``.
        """
        callable = cast(Callable[[Event], None], callable)
        try:
            self._subscriptions.remove((event_type, callable))
        except ValueError:
            name = "all events" if event_type is None else event_type.__name__
            raise ValueError(f"{callable!r} is not subscribed to {name}") from None

    def notify(self, event: Event) -> None:
        """Call every callback subscribed to the event's class or a parent class.

        Callbacks run in the order they were subscribed. A callback subscribed to
        several matching classes runs once for each.
        """
        # Copy, so a callback can subscribe or unsubscribe while this runs.
        for event_type, callback in list(self._subscriptions):
            if event_type is None or isinstance(event, event_type):
                callback(event)


class Handler(ABC):
    """Abstract interface for components that consume events.

    A handler holds its subscription logic and can attach itself to a
    :py:class:`Dispatcher`.

    Subclass ``Handler`` and implement :py:meth:`attach_with` to register
    the handler's callbacks with a dispatcher.
    """

    @abstractmethod
    def attach_with(self, dispatcher: Dispatcher) -> "Handler":
        """Attach this handler to a dispatcher.

        Implementations should call :py:meth:`capymoa.events.Dispatcher.subscribe`
        for each event type the handler needs.

        :param dispatcher: The dispatcher to subscribe to.
        """
        return self


__all__ = ["Dispatcher", "Event", "Handler"]
