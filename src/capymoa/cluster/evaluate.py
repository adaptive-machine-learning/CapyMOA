"""Evaluator for clustering."""

from capymoa.base import Clusterer


class ClusteringEvaluator:
    """
    Abstract clustering evaluator for CapyMOA.
    It is slightly different from the other evaluators because it does not have a moa_evaluator object.
    Clustering evaluation at this point is very simple and only uses the unsupervised metrics.
    """

    def __init__(self, update_interval=1000):
        """
        Only the update_interval is set here.
        :param update_interval: The interval at which the evaluator should update the measurements
        """
        self.instances_seen = 0
        self.update_interval = update_interval
        self.measurements = {name: [] for name in self.metrics_header()}
        self.clusterer_name = None

    def __str__(self):
        return str(self.metrics_dict())

    def get_instances_seen(self):
        return self.instances_seen

    def get_update_interval(self):
        return self.update_interval

    def get_clusterer_name(self):
        return self.clusterer_name

    def update(self, clusterer: Clusterer):
        if self.clusterer_name is None:
            self.clusterer_name = str(clusterer)
        self.instances_seen += 1
        if self.instances_seen % self.update_interval == 0:
            self._update_measurements(clusterer)

    def _update_measurements(self, clusterer: Clusterer):
        # update centers, weights, sizes, and radii
        if clusterer.implements_macro_clusters():
            macro = clusterer.get_clustering_result()
            if len(macro.get_centers()) > 0:
                self.measurements["macro"].append(macro)

        if clusterer.implements_micro_clusters():
            micro = clusterer.get_micro_clustering_result()
            if len(micro.get_centers()) > 0:
                self.measurements["micro"].append(micro)

        # calculate silhouette score
        # TODO: delegate silhouette to moa
        # Check how it is done among different clusterers

    def metrics_header(self):
        performance_names = ["macro", "micro"]
        return performance_names

    def metrics(self):
        # using the static list to keep the order of the metrics
        return [self.measurements[key] for key in self.metrics_header()]

    def get_measurements(self):
        return self.measurements
