from abc import ABC, abstractmethod
from typing import Callable, Dict, Type, TypeVar, cast


class Event:
    """Base class for events emitted by a :py:class:`Dispatcher`."""


_E = TypeVar("_E", bound=Event)


class Dispatcher:
    """Publish events to subscribed callbacks.

    Create a source, subscribe one or more handlers for an event type, and
    dispatch event instances with :py:meth:`notify`.

    >>> source = Dispatcher()
    >>> def handler(event: Event): print(f"Received event: {type(event).__name__}")
    >>> source.subscribe(Event, handler)
    >>> source.notify(Event()) # doctest: +ELLIPSIS
    Received event: Event

    Subscriptions are keyed by exact event class.
    """

    def __init__(self) -> None:
        """Initialize an empty mapping of event types to callbacks."""
        self.subscribers: Dict[Type[Event] | None, list[Callable[[Event], None]]] = {}

    def subscribe(
        self, event_type: Type[_E] | None, callable: Callable[[_E], None]
    ) -> None:
        """Register a callback for a specific event class."""
        callable = cast(Callable[[Event], None], callable)
        self.subscribers.setdefault(event_type, []).append(callable)

    def unsubscribe(
        self, event_type: Type[_E] | None, callable: Callable[[_E], None]
    ) -> None:
        """Remove a previously registered callback for an event class."""
        callable = cast(Callable[[Event], None], callable)
        self.subscribers.get(event_type, []).remove(callable)

    def notify(self, event: Event) -> None:
        """Notify all subscribers of an event.

        Only subscribers of the exact event type will be notified. Subscribers of parent
        event types will not be notified.
        """
        for subscriber in self.subscribers.get(type(event), []):
            subscriber(event)
        for subscriber in self.subscribers.get(None, []):
            subscriber(event)


class Handler(ABC):
    """Abstract interface for components that consume events.

    A sink encapsulates subscription logic and can attach itself to an
    :py:class:`Dispatcher`.

    Subclass ``Handler`` and implement :py:meth:`attach_with` to register
    the sink's handlers with a source.
    """

    @abstractmethod
    def attach_with(self, dispatcher: Dispatcher) -> "Handler":
        """Attach this sink to an event source.

        Implementations should call
        :py:meth:`capymoa.ocl.events.Dispatcher.subscribe` for each
        event type the sink needs to handle.

        :param dispatcher: The source this sink should subscribe to.
        """
        return self


__all__ = ["Event", "Dispatcher", "Handler"]
