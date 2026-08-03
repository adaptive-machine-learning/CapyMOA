from abc import ABC, abstractmethod
from typing import Any, Dict, Type
from typing_extensions import override
from moa.classifiers.core.driftdetection import (
    AbstractChangeDetector as _AbstractChangeDetector,
)


class BaseDriftDetector(ABC):
    """Drift Detector"""

    def __init__(self):
        super().__init__()

        self.in_concept_change = None
        self.in_warning_zone = None
        self.detection_index = []
        self.warning_index = []
        self.data = []
        self.idx = 0

    @abstractmethod
    def get_params(self) -> Dict[str, Any]:
        """Get the hyper-parameters of the drift detector."""

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

    _moa_detector_type: Type[_AbstractChangeDetector] | None = None

    def __init__(
        self,
        cli: str = "",
        moa_detector_type: Type[_AbstractChangeDetector] | None = None,
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

    @override
    def get_params(self) -> Dict[str, Any]:
        options = list(self.moa_detector.getOptions().getOptionArray())
        return {opt.getName(): opt.getValueAsCLIString() for opt in options}

    def cli_help(self) -> str:
        return str(self.moa_detector.getOptions().getHelpString())

    def __str__(self) -> str:
        full_name = str(self.moa_detector.getClass().getCanonicalName())
        return full_name.rsplit(".", 1)[1] if "." in full_name else full_name
