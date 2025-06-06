from typing import Union, List
from typing_extensions import override

import numpy as np

from capymoa.base import MOAClassifier
from capymoa.drift.detectors import ADWIN
from capymoa.drift.base_detector import BaseDriftDetector, MOADriftDetector
from capymoa.instance import LabeledInstance, Instance

ArrayOrList = Union[np.ndarray, List[float]]
ArrayOrInstance = Union[ArrayOrList, Instance]


class STUDD(BaseDriftDetector):
    """STUDD: Student-Teacher Unsupervised Drift Detection

    STUDD is a concept drift detection method that uses a student-teacher approach.
    It trains a student model to mimic a teacher model's predictions and monitors
    the agreement between them to detect concept drift in an unsupervised manner.

    The detector works by:
    1. Training a student model on the same data as the teacher
    2. Monitoring the agreement between student and teacher predictions
    3. Using a base drift detector (e.g., ADWIN) on the agreement signal

    Example usage:
    -------------
    >>> from capymoa.drift.detectors import ADWIN
    >>> from capymoa.drift.detectors import STUDD
    >>> from capymoa.classifier import AdaptiveRandomForestClassifier as ARF
    >>> from capymoa.datasets import ElectricityTiny
    >>>
    >>> stream = ElectricityTiny()
    >>>
    >>> learner = ARF(schema=stream.get_schema())
    >>> student = ARF(schema=stream.get_schema())
    >>>
    >>> detector = STUDD(student=student, detector=ADWIN())
    >>>
    >>> instances_processed = 0
    >>> while stream.has_more_instances():
    >>>     instance = stream.next_instance()
    >>>
    >>>     prediction = learner.predict(instance)
    >>>     detector.add_element(instance, prediction)
    >>>
    >>>     if detector.detected_change():
    >>>         print(f'Change detected at index: {instances_processed}')
    >>>
    >>>     instances_processed += 1

    Reference:
    ----------
    Cerqueira, V., Gomes, H. M., Bifet, A., & Torgo, L. (2023).
    STUDD: A student–teacher method for unsupervised concept drift detection.
    Machine Learning, 112(11), 4351-4378.

    """

    def __init__(
        self,
        student: MOAClassifier,
        min_n_instances: int = 500,
        detector: MOADriftDetector = ADWIN(),
    ):
        """
        :param student: Student model that mimics the teacher's predictions
        :param min_n_instances: Minimum number of instances before monitoring drift
        :param detector: Base drift detector to monitor agreement signal
        """
        super().__init__()

        self.min_n_instances = min_n_instances
        self.detector = detector
        self.student = student

        self.in_concept_change = self.detector.in_concept_change
        self.in_warning_zone = self.detector.in_warning_zone
        self.detection_index = self.detector.detection_index
        self.warning_index = self.detector.warning_index
        self.data = self.detector.data

    def __str__(self):
        return "STUDD"

    @override
    def add_element(self, instance_x: ArrayOrInstance, teacher_prediction) -> None:
        """Update the drift detector with a new instance and teacher prediction.

        Parameters
        ----------
        instance_x : Instance
            The instance to add to the drift detector.
        teacher_prediction : Any
            The prediction made by the teacher model for this instance.
        """
        self.idx += 1

        if hasattr(instance_x, "x"):
            # class Instance
            tmp_instance = self.instance_from_arr(instance_x.x, teacher_prediction)
        else:
            tmp_instance = self.instance_from_arr(instance_x, teacher_prediction)

        # Only detect drift after seeing minimum number of instances
        # important to have a minimum number of instances to train the student model
        if self.idx >= self.min_n_instances:
            pred = self.student.predict(tmp_instance)

            meta_y = int(pred == teacher_prediction)

            self.detector.add_element(meta_y)

            self.in_concept_change = self.detector.in_concept_change
            self.in_warning_zone = self.detector.in_warning_zone
            self.detection_index = self.detector.detection_index
            self.warning_index = self.detector.warning_index
            self.data = self.detector.data

        self.student.train(tmp_instance)

    def instance_from_arr(self, x, y):
        """Convert an instance and prediction to a LabeledInstance.

        Parameters
        ----------
        x : Any
            The instance to convert.
        y : Any
            The prediction made by the teacher model for this instance.

        Returns
        -------
        tmp_instance : LabeledInstance
            The converted LabeledInstance.
        """
        tmp_instance = LabeledInstance.from_array(self.student.schema, x, y)
        return tmp_instance

    @override
    def get_params(self):
        """Get the parameters of the drift detector.

        Returns
        -------
        params : dict
            Dictionary containing the detector parameters, including
            min_n_instances and information about the student and detector.
        """
        params = self.detector.get_params()
        params.update(
            {
                "min_n_instances": self.min_n_instances,
                "student": str(self.student),
                "detector_type": str(self.detector),
            }
        )

        return params

    @override
    def reset(self, clean_history: bool = False) -> None:
        """Reset the drift detector.

        :param clean_history: Whether to reset detection history, defaults to False
        """
        self.detector.reset(clean_history=clean_history)

        self.in_concept_change = self.detector.in_concept_change
        self.in_warning_zone = self.detector.in_warning_zone

        if clean_history:
            self.detection_index = self.detector.detection_index
            self.warning_index = self.detector.warning_index
            self.data = self.detector.data
            self.idx = 0
