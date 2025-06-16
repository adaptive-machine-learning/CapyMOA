from abc import ABC, abstractmethod
# from typing import Optional
from capymoa.instance import Instance
# , LabeledInstance, RegressionInstance
from capymoa.stream._stream import Schema
# from capymoa.type_alias import LabelIndex, LabelProbabilities, TargetValue
from capymoa.cluster.results import ClusteringResult

class Cluster(ABC):
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

    @abstractmethod
    def _is_visualization_supported(self):
        pass

    # @abstractmethod
    # def predict(self, instance: Instance) -> Optional[LabelIndex]:
    #     pass

    # @abstractmethod
    # def predict_proba(self, instance: Instance) -> LabelProbabilities:
    #     pass


class MOACluster(Cluster):
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

    def CLI_help(self):
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
        result = self.moa_learner.getClusteringResult()
        if result is None:
            return ret
        clustering = result.getClustering()
        if clustering is None or clustering.size() == 0:
            return ret
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
            centers = self._get_clusters_centers()
            weights = self._get_clusters_weights()
            radii = self._get_clusters_radii()
            # raise ValueError("This clusterer does not implement macro-clusters.")
            return ClusteringResult(
                centers,
                weights,
                radii,
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
