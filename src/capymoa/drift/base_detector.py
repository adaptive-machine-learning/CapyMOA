from abc import ABC, abstractmethod
from typing import Any, Dict
from typing_extensions import override

from jpype import _jpype


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
    """
    A wrapper class for using MOA (Massive Online Analysis) drift detectors in CapyMOA.
    """

    def __init__(self, moa_detector, CLI=None):
        """
        :param moa_detector: The MOA detector object or class identifier.
        :param CLI: The command-line interface (CLI) configuration for the MOA drift detector, defaults to None
        """
        super().__init__()

        self.CLI = CLI

        if isinstance(moa_detector, type):
            if isinstance(moa_detector, _jpype._JClass):
                moa_detector = moa_detector()
            else:  # this is not a Java object, thus it certainly isn't a MOA learner
                raise ValueError("Invalid MOA detector provided.")

        self.moa_detector = moa_detector

        # If the CLI is None, we assume the object has already been configured
        # or that default values should be used.
        if self.CLI is not None:
            self.moa_detector.getOptions().setViaCLIString(CLI)

        self.moa_detector.prepareForUse()
        self.moa_detector.resetLearning()

    def __str__(self):
        full_name = str(self.moa_detector.getClass().getCanonicalName())
        return full_name.rsplit(".", 1)[1] if "." in full_name else full_name

    def CLI_help(self):
        return str(self.moa_detector.getOptions().getHelpString())

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

    @override
    def get_params(self) -> Dict[str, Any]:
        options = list(self.moa_detector.getOptions().getOptionArray())
        return {opt.getName(): opt.getValueAsCLIString() for opt in options}
