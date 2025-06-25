from abc import ABC, abstractmethod
from typing import Optional

from jpype import _jpype
from moa.classifiers import (
    Classifier as MOA_Classifier_Interface,
)
from moa.classifiers import (
    Regressor as MOA_Regressor_Interface,
)
from moa.classifiers.predictioninterval import (
    PredictionIntervalLearner as MOA_PredictionInterval_Interface,
)
from moa.classifiers.trees import (
    ARFFIMTDD as MOA_ARFFIMTDD,
)
from moa.classifiers.trees import (
    SelfOptimisingBaseTree as MOA_SOKNLBT,
)
from moa.core import Utils

from capymoa.base._classifier import MOAClassifier
from capymoa.base._regressor import MOARegressor, Regressor
from capymoa.instance import Instance
from capymoa.stream._stream import Schema
from capymoa.type_alias import LabelIndex

##############################################################
##################### INTERNAL FUNCTIONS #####################
##############################################################


def _extract_moa_drift_detector_CLI(drift_detector):
    """
    Auxiliary function to retrieve the command-line interface (CLI)
    creation command for a MOA Drift Detector.

    Parameters:
    - drift_detector: The drift detector class for which the CLI command is needed

    Returns:
    A string representing the CLI command for creating a Drift Detector.
    """

    CLI = drift_detector.CLI

    moa_detector = drift_detector.moa_detector
    moa_detector_class_id = str(moa_detector.getClass().getName())
    moa_detector_class_id_parts = moa_detector_class_id.split(".")

    moa_detector_str = f"{moa_detector_class_id_parts[-1]}"
    moa_detector_str = f"({moa_detector_str} {CLI})"

    return moa_detector_str


def _get_moa_creation_CLI(moa_learner):
    """
    Auxiliary function to retrieve the command-line interface (CLI)
    creation command for a MOA learner.

    Parameters:
    - moa_learner: The MOA learner for which the CLI command is needed,
    it must be a MOA learner object

    Returns:
    A string representing the CLI command for creating the MOA learner.
    """
    moa_learner_class_id = str(moa_learner.getClass().getName())
    moa_learner_class_id_parts = moa_learner_class_id.split(".")

    moa_learner_str = (
        f"{moa_learner_class_id_parts[-1]}"
        if isinstance(moa_learner, MOA_ARFFIMTDD)
        or isinstance(moa_learner, MOA_SOKNLBT)
        else f"{moa_learner_class_id_parts[-2]}.{moa_learner_class_id_parts[-1]}"
    )

    moa_cli_creation = str(moa_learner.getCLICreationString(moa_learner.__class__))
    CLI = moa_cli_creation.split(" ", 1)

    if len(CLI) > 1 and len(CLI[1]) > 1:
        moa_learner_str = f"({moa_learner_str} {CLI[1]})"

    return moa_learner_str


def _extract_moa_learner_CLI(learner):
    """
    Auxiliary function to extract the command-line interface (CLI)
    from a MOA learner object or a MOA learner class or even a
    MOAClassifier object (which has a moa_learner internally).

    Parameters:
    - moa_learner: The object or class representing the MOA learner

    Returns:
    A string representing the CLI command for creating the MOA learner.
    """

    # Check if the base_learner is a MOAClassifie or a MOARegressor
    if (
        isinstance(learner, MOAClassifier)
        or isinstance(learner, MOARegressor)
        or isinstance(learner, MOAPredictionIntervalLearner)
    ):
        learner = _get_moa_creation_CLI(learner.moa_learner)

    # ... or a Classifier or a Regressor (Interfaces from MOA) type
    if (
        isinstance(learner, MOA_Classifier_Interface)
        or isinstance(learner, MOA_Regressor_Interface)
        or isinstance(learner, MOA_PredictionInterval_Interface)
    ):
        learner = _get_moa_creation_CLI(learner)

    # ... or a java object, which we presume is a MOA object (if it is not, MOA will raise the error)
    if isinstance(learner, _jpype._JClass):
        learner = _get_moa_creation_CLI(learner())
    return learner


##############################################################
######################### REGRESSORS #########################
##############################################################


### Prediction Interval Learner ###
class PredictionIntervalLearner(Regressor):
    def __init__(self, schema=None, random_seed=1):
        super().__init__(schema=schema, random_seed=random_seed)

    @abstractmethod
    def train(self, instance):
        pass

    @abstractmethod
    def predict(self, instance):
        pass


class MOAPredictionIntervalLearner(MOARegressor, PredictionIntervalLearner):
    def train(self, instance):
        self.moa_learner.trainOnInstance(instance.java_instance)

    def predict(self, instance):
        prediction_PI = self.moa_learner.getVotesForInstance(instance.java_instance)
        if len(prediction_PI) != 3:
            return [0, 0, 0]
        else:
            return prediction_PI


class AnomalyDetector(ABC):
    """
    Abstract base class for anomaly detector.

    Attributes:
    - schema: The schema representing the instances. Defaults to None.
    - random_seed: The random seed for reproducibility. Defaults to 1.
    """

    def __init__(self, schema: Schema, random_seed=1):
        self.random_seed = random_seed
        self.schema = schema
        if self.schema is None:
            raise ValueError("Schema must be initialised")

    def __str__(self):
        pass

    @abstractmethod
    def train(self, instance: Instance):
        pass

    @abstractmethod
    def predict(self, instance: Instance) -> Optional[LabelIndex]:
        # Returns the predicted label for the instance.
        pass

    @abstractmethod
    def score_instance(self, instance: Instance) -> float:
        """Returns the anomaly score for the instance.

        A high score is indicative of an anomaly.

        :param instance: The instance for which the anomaly score is calculated.
        :return: The anomaly score for the instance.
        """


class MOAAnomalyDetector(AnomalyDetector):
    def __init__(self, schema=None, CLI=None, random_seed=1, moa_learner=None):
        super().__init__(schema=schema, random_seed=random_seed)
        self.CLI = CLI
        self.moa_learner = moa_learner

        if random_seed is not None:
            self.moa_learner.setRandomSeed(random_seed)

        if self.schema is not None:
            self.moa_learner.setModelContext(self.schema.get_moa_header())

        if self.CLI is not None:
            self.moa_learner.getOptions().setViaCLIString(CLI)

        self.moa_learner.prepareForUse()
        self.moa_learner.resetLearning()
        self.moa_learner.setModelContext(self.schema.get_moa_header())

    def __str__(self):
        full_name = str(self.moa_learner.getClass().getCanonicalName())
        return full_name.rsplit(".", 1)[1] if "." in full_name else full_name

    def cli_help(self):
        return self.moa_learner.getOptions().getHelpString()

    def train(self, instance):
        self.moa_learner.trainOnInstance(instance.java_instance)

    def predict(self, instance):
        return Utils.maxIndex(
            self.moa_learner.getVotesForInstance(instance.java_instance)
        )

    def score_instance(self, instance):
        # We assume that the anomaly score is the second element of the prediction array.
        # However, if it is not the case for a MOA learner, this method should be overridden.
        prediction_array = self.moa_learner.getVotesForInstance(instance.java_instance)
        return float(prediction_array[1])


##############################################################
######################### Clustering #########################
##############################################################
class ClusteringResult:
    """Abstract clustering result class that has the structure of clusters: centers, weights, radii, and ids.

    IDs might not be available for most MOA implementations."""

    def __init__(self, centers, weights, radii, ids):
        self._centers = centers
        self._weights = weights
        self._radii = radii
        self._ids = ids

    def get_centers(self):
        return self._centers

    def get_weights(self):
        return self._weights

    def get_radii(self):
        return self._radii

    def get_ids(self):
        return self._ids

    def __str__(self) -> str:
        return f"Centers: {self._centers}, Weights: {self._weights}, Radii: {self._radii}, IDs: {self._ids}"


class Clusterer(ABC):
    def __init__(self, schema: Schema, random_seed=1):
        self.random_seed = random_seed
        self.schema = schema
        if self.schema is None:
            raise ValueError("Schema must be initialised")

    @abstractmethod
    def __str__(self):
        pass

    @abstractmethod
    def train(self, instance: Instance):
        pass

    @abstractmethod
    def implements_micro_clusters(self) -> bool:
        pass

    @abstractmethod
    def implements_macro_clusters(self) -> bool:
        pass

    @abstractmethod
    def _get_micro_clusters_centers(self):
        pass

    @abstractmethod
    def _get_micro_clusters_radii(self):
        pass

    @abstractmethod
    def _get_micro_clusters_weights(self):
        pass

    @abstractmethod
    def _get_clusters_centers(self):
        pass

    @abstractmethod
    def _get_clusters_radii(self):
        pass

    @abstractmethod
    def _get_clusters_weights(self):
        pass

    @abstractmethod
    def get_clustering_result(self):
        pass

    @abstractmethod
    def get_micro_clustering_result(self):
        pass

    # @abstractmethod
    # def predict(self, instance: Instance) -> Optional[LabelIndex]:
    #     pass

    # @abstractmethod
    # def predict_proba(self, instance: Instance) -> LabelProbabilities:
    #     pass


class MOAClusterer(Clusterer):
    """
    A wrapper class for using MOA (Massive Online Analysis) clusterers in CapyMOA.

    Attributes:
    - schema: The schema representing the instances. Defaults to None.
    - CLI: The command-line interface (CLI) configuration for the MOA learner.
    - random_seed: The random seed for reproducibility. Defaults to 1.
    - moa_learner: The MOA learner object or class identifier.
    """

    def __init__(self, moa_learner, schema=None, CLI=None):
        super().__init__(schema=schema)
        self.CLI = CLI
        # If moa_learner is a class identifier instead of an object
        if isinstance(moa_learner, type):
            if isinstance(moa_learner, _jpype._JClass):
                moa_learner = moa_learner()
            else:  # this is not a Java object, thus it certainly isn't a MOA learner
                raise ValueError("Invalid MOA clusterer provided.")
        self.moa_learner = moa_learner

        # self.moa_learner.setRandomSeed(self.random_seed)

        if self.schema is not None:
            self.moa_learner.setModelContext(self.schema.get_moa_header())

        # If the CLI is None, we assume the object has already been configured
        # or that default values should be used.
        if self.CLI is not None:
            self.moa_learner.getOptions().setViaCLIString(CLI)

        self.moa_learner.prepareForUse()
        self.moa_learner.resetLearningImpl()
        self.moa_learner.setModelContext(schema.get_moa_header())

    def __str__(self):
        # Removes the package information from the name of the learner.
        full_name = str(self.moa_learner.getClass().getCanonicalName())
        return full_name.rsplit(".", 1)[1] if "." in full_name else full_name

    def cli_help(self):
        return str(self.moa_learner.getOptions().getHelpString())

    def train(self, instance):
        self.moa_learner.trainOnInstance(instance.java_instance.getData())

    def _get_micro_clusters_centers(self):
        ret = []
        for c in self.moa_learner.getMicroClusteringResult().getClustering():
            java_array = c.getCenter()[:-1]
            python_array = [
                java_array[i] for i in range(len(java_array))
            ]  # Convert to Python list
            ret.append(python_array)
        return ret

    def _get_micro_clusters_radii(self):
        ret = []
        for c in self.moa_learner.getMicroClusteringResult().getClustering():
            ret.append(c.getRadius())
        return ret

    def _get_micro_clusters_weights(self):
        ret = []
        for c in self.moa_learner.getMicroClusteringResult().getClustering():
            ret.append(c.getWeight())
        return ret

    def _get_clusters_centers(self):
        ret = []
        for c in self.moa_learner.getClusteringResult().getClustering():
            java_array = c.getCenter()[:-1]
            python_array = [
                java_array[i] for i in range(len(java_array))
            ]  # Convert to Python list
            ret.append(python_array)
        return ret

    def _get_clusters_radii(self):
        ret = []
        for c in self.moa_learner.getClusteringResult().getClustering():
            ret.append(c.getRadius())
        return ret

    def _get_clusters_weights(self):
        ret = []
        for c in self.moa_learner.getClusteringResult().getClustering():
            ret.append(c.getWeight())
        return ret

    def get_clustering_result(self):
        if self.implements_macro_clusters():
            # raise ValueError("This clusterer does not implement macro-clusters.")
            return ClusteringResult(
                self._get_clusters_centers(),
                self._get_clusters_weights(),
                self._get_clusters_radii(),
                [],
            )
        else:
            return ClusteringResult([], [], [], [])

    def get_micro_clustering_result(self):
        if self.implements_micro_clusters():
            return ClusteringResult(
                self._get_micro_clusters_centers(),
                self._get_micro_clusters_weights(),
                self._get_micro_clusters_radii(),
                [],
            )
        else:
            return ClusteringResult([], [], [], [])

    # def predict(self, instance):
    #     return Utils.maxIndex(
    #         self.moa_learner.getVotesForInstance(instance.java_instance)
    #     )

    # def predict_proba(self, instance):
    #     return self.moa_learner.getVotesForInstance(instance.java_instance)
