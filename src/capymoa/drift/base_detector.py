import functools
from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import Any

from moa.classifiers.core.driftdetection import (
    AbstractChangeDetector as _AbstractChangeDetector,
)
from typing_extensions import override

from capymoa.base._learner_params import LearnerParamsMixin
from capymoa.drift.events import DriftEvent, WarningEvent
from capymoa.events import Dispatcher, Event


def _emit_after(add_element: Callable[..., Any]) -> Callable[..., Any]:
    """Wrap ``add_element`` so the detector emits events after it runs.

    An event is emitted for each entry ``add_element`` adds to
    ``warning_index`` or ``detection_index``. The history is used rather than
    ``in_warning_zone`` and ``in_concept_change``, because some detectors leave
    those flags unchanged when they return early.
    """

    @functools.wraps(add_element)
    def wrapper(self: "BaseDriftDetector", *args: Any, **kwargs: Any) -> Any:
        # A subclass calling super().add_element() must not emit twice.
        if getattr(self, "_adding_element", False):
            return add_element(self, *args, **kwargs)

        n_warnings, n_drifts = len(self.warning_index), len(self.detection_index)
        self._adding_element = True
        try:
            result = add_element(self, *args, **kwargs)
        finally:
            self._adding_element = False

        for index in self.warning_index[n_warnings:]:
            self.events.notify(WarningEvent(self, index))
        for index in self.detection_index[n_drifts:]:
            self.events.notify(DriftEvent(self, index))
        return result

    return wrapper


class BaseDriftDetector(LearnerParamsMixin, ABC):
    """Drift Detector

    A detector emits a :class:`~capymoa.drift.events.WarningEvent` or a
    :class:`~capymoa.drift.events.DriftEvent` each time it records a warning or
    a drift. Subscribe through :attr:`events`, and see :mod:`capymoa.drift.events`
    for an example. :attr:`emits` lists the events a detector can emit.
    """

    emits: tuple[type[Event], ...] = (WarningEvent, DriftEvent)
    """The events this detector emits through :attr:`events`."""

    REQUIRES_FIT: bool = False
    """If ``True``, this detector needs a reference distribution before
    it can detect change. Concept drift detectors leave this as
    ``False`` and only use :meth:`add_element`."""

    def __init_subclass__(cls, **kwargs) -> None:
        super().__init_subclass__(**kwargs)
        add_element = cls.__dict__.get("add_element")
        if add_element is not None and not getattr(
            add_element, "__isabstractmethod__", False
        ):
            cls.add_element = _emit_after(add_element)

    def __init__(self):
        super().__init__()

        self.events = Dispatcher(emits=self.emits)
        """Subscribe here to be called on a warning or a drift."""
        self.in_concept_change = None
        self.in_warning_zone = None
        self.detection_index = []
        self.warning_index = []
        self.data = []
        self.idx = 0

    def reset(self, clean_history: bool = False) -> None:
        """Reset the drift detector.

        :param clean_history: Whether to reset detection history, defaults to False
        """
        self.in_concept_change = False
        self.in_warning_zone = False

        if clean_history:
            self.detection_index = []
            self.warning_index = []
            self.data = []
            self.idx = 0

    @abstractmethod
    def add_element(self, element: float) -> None:
        """Update the drift detector with a new input value.

        :param element: A value to update the drift detector with. Usually,
            this is the prediction error of a model.
        """
        raise NotImplementedError

    def detected_change(self) -> bool:
        """Is the detector currently detecting a concept drift?"""
        return self.in_concept_change

    def detected_warning(self) -> bool:
        """Is the detector currently warning of an upcoming concept drift?"""
        return self.in_warning_zone


class MOADriftDetector(BaseDriftDetector):
    """A MOA (Massive Online Analysis) drift detector for CapyMOA."""

    _moa_detector_type: type[_AbstractChangeDetector] | None = None

    def __init__(
        self,
        cli: str = "",
        moa_detector_type: type[_AbstractChangeDetector] | None = None,
    ):
        """Initialize the wrapped MOA drift detector."""
        super().__init__()
        # Allow passing the MOA detector type directly to the constructor, which is
        # useful for the from_cli class method. Otherwise, use the class attribute
        # _moa_detector_type defined in each subclass.
        if moa_detector_type is not None:
            self._moa_detector_type = moa_detector_type
        if self._moa_detector_type is None:
            raise NotImplementedError(
                "MOA detector type not specified. Set the class attribute "
                "_moa_detector_type or pass moa_detector_type to the constructor."
            )

        # Setup Detector
        self.moa_detector = self._moa_detector_type()
        self.moa_detector.getOptions().setViaCLIString(cli)
        self.moa_detector.prepareForUse()
        self.moa_detector.resetLearning()

    @classmethod
    def from_cli(cls, cli: str) -> "MOADriftDetector":
        """Create a detector instance configured from a MOA CLI string.

        :param cli: Command-line style options string for MOA detector hyper-parameters.
        :return: A new detector instance initialized with ``cli``.
        """
        if cls._moa_detector_type is None:
            raise NotImplementedError("Unset class attribute _moa_detector_type.")

        instance = cls.__new__(cls)
        MOADriftDetector.__init__(
            instance, cli=cli, moa_detector_type=cls._moa_detector_type
        )
        return instance

    @override
    def add_element(self, element: float) -> None:
        self.moa_detector.input(element)
        self.data.append(element)
        self.idx += 1

        self.in_concept_change = self.moa_detector.getChange()
        self.in_warning_zone = self.moa_detector.getWarningZone()

        if self.in_warning_zone:
            self.warning_index.append(self.idx)

        if self.in_concept_change:
            self.detection_index.append(self.idx)

    def reset(self, clean_history: bool = False) -> None:
        """Reset the drift detector.

        :param clean_history: Whether to reset detection history, defaults to False
        """
        self.in_concept_change = False
        self.in_warning_zone = False
        self.moa_detector.prepareForUse()
        self.moa_detector.resetLearning()

        if clean_history:
            self.detection_index = []
            self.warning_index = []
            self.data = []
            self.idx = 0

    def cli_help(self) -> str:
        return str(self.moa_detector.getOptions().getHelpString())

    def __str__(self) -> str:
        full_name = str(self.moa_detector.getClass().getCanonicalName())
        return full_name.rsplit(".", 1)[1] if "." in full_name else full_name
