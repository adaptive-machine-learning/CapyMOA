from abc import ABC, abstractmethod

from jpype import _jpype


class BaseDriftDetector(ABC):
    """ Drift Detector

    """

    def __init__(self):
        super().__init__()

        self.in_concept_change = None
        self.in_warning_zone = None
        self.estimation = None
        self.detection_index = []
        self.warning_index = []
        self.data = []
        self.idx = 0

        self.params = {}

    def get_params(self):
        pass

    def reset(self, clean_history: bool = False):
        """ reset

        Resets the drift detector

        Parameters
        ----------
        clean_history: bool
            Whether to reset detection history

        Returns
        -------
        BaseDriftDetector
            self, optional


        """
        self.in_concept_change = False
        self.in_warning_zone = False
        self.estimation = 0.0

        if clean_history:
            self.detection_index = []
            self.warning_index = []
            self.data = []
            self.idx = 0

    @abstractmethod
    def add_element(self, input_value):
        """ add_element

        Adds the relevant data from a sample into the change detector.

        Parameters
        ----------
        input_value: Not defined
            Whatever input value the change detector takes.

        Returns
        -------
        BaseDriftDetector
            self, optional

        """
        raise NotImplementedError

    def detected_change(self):
        """ detected_change

        This function returns whether concept drift was detected or not.

        Returns
        -------
        bool
            Whether concept drift was detected or not.

        """
        return self.in_concept_change

    def detected_warning(self):
        """ detected_warning_zone

        If the change detector supports the warning zone, this function will return
        whether it's inside the warning zone or not.

        Returns
        -------
        bool
            Whether the change detector is in the warning zone or not.

        """
        return self.in_warning_zone

    def get_estimation(self):
        """ get_estimation

        Returns the detector estimation.

        Returns
        -------
        float
            estimated value by the detector

        """
        return self.estimation


class MOADriftDetector(BaseDriftDetector):
    """
    A wrapper class for using MOA (Massive Online Analysis) drift detectors in CapyMOA.

    Attributes:
    - moa_detector: The MOA detector object or class identifier.
    - CLI: The command-line interface (CLI) configuration for the MOA drift detector.
    """

    def __init__(self, moa_detector, CLI=None):
        super().__init__()

        self.CLI = CLI

        if isinstance(moa_detector, type):
            if type(moa_detector) == _jpype._JClass:
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

    def add_element(self, element):
        self.moa_detector.input(element)
        self.data.append(element)
        self.idx += 1

        self.estimation = self.moa_detector.getEstimation()
        self.in_concept_change = self.moa_detector.getChange()
        self.in_warning_zone = self.moa_detector.getWarningZone()

        if self.in_warning_zone:
            self.warning_index.append(self.idx)

        if self.in_concept_change:
            self.detection_index.append(self.idx)

    def get_estimation(self):
        return self.estimation

    def detected_change(self):
        return self.in_concept_change

    def detected_warning(self):
        return self.in_warning_zone

    def get_params(self):
        options = list(self.moa_detector.getOptions().getOptionArray())
        self.params = {opt.getName(): opt.getValueAsCLIString() for opt in options}
        return self.params

    def reset(self, clean_history: bool = False):
        self.in_concept_change = False
        self.in_warning_zone = False
        self.estimation = 0.0
        self.moa_detector.resetLearning()

        if clean_history:
            self.detection_index = []
            self.warning_index = []
            self.data = []
            self.idx = 0
