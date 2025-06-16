from capymoa.cluster.base import Cluster
from capymoa.cluster.results import PrequentialClusteringResults
from capymoa.evaluation.evaluation import start_time_measuring, stop_time_measuring
import numpy as np
from sklearn.metrics import pairwise_distances


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
        self.cluster_name = None

    def __str__(self):
        return str(self.metrics_dict())

    def get_instances_seen(self):
        return self.instances_seen

    def get_update_interval(self):
        return self.update_interval

    def get_cluster_name(self):
        return self.cluster_name

    def _silhouette_score(self, cluster: Cluster, past_dps=None, future_dps=None):
        if cluster.implements_macro_clusters():
            centers = np.array(cluster._get_clusters_centers())
        else:
            centers = np.array(cluster._get_micro_clusters_centers())

        if centers.size == 0 or centers is None:
            return [0, 0]

        silhouette_scores = []
        # Assign each data point to the nearest cluster center
        if past_dps is not None and future_dps is not None:
            for _ in [past_dps, future_dps]:
                labels = []
                for point in _:
                    distances = np.linalg.norm(centers - point.x, axis=1)
                    labels.append(np.argmin(distances))

                labels = np.array(labels)

                # Compute pairwise distances between data points
                xs = np.array([p.x for p in _])
                distance_matrix = pairwise_distances(xs)

                silhouette_values = []
                for i, point in enumerate(_):
                    cluster_label = labels[i]

                    # Intra-cluster distance (a): Average distance to other points in the same cluster
                    in_cluster_points = [j for j in range(len(_)) if labels[j] == cluster_label and j != i]
                    a = np.mean([distance_matrix[i, j] for j in in_cluster_points]) if in_cluster_points else 0

                    # Inter-cluster distance (b): Minimum average distance to points in the nearest other cluster
                    other_clusters = set(labels) - {cluster_label}
                    b_values = []

                    for other_cluster in other_clusters:
                        other_cluster_points = [j for j in range(len(_)) if labels[j] == other_cluster]
                        if other_cluster_points:
                            b_values.append(np.mean([distance_matrix[i, j] for j in other_cluster_points]))

                    b = min(b_values) if b_values else 0

                    # Silhouette score for this point
                    s = (b - a) / max(a, b) if max(a, b) > 0 else 0
                    silhouette_values.append(s)

                # Overall silhouette score
                silhouette_scores.append(np.mean(silhouette_values) if silhouette_values else 0)
        return silhouette_scores
    
    def _ssq_score(self, cluster: Cluster, past_dps=None, future_dps=None):
        if cluster.implements_macro_clusters():
            centers = np.array(cluster._get_clusters_centers())
        else:
            centers = np.array(cluster._get_micro_clusters_centers())

        if centers.size == 0 or centers is None:
            return [0, 0]

        ssq_scores = []
        # Assign each data point to the nearest cluster center
        if past_dps is not None and future_dps is not None:
            for _ in [past_dps, future_dps]:
                labels = []
                for point in _:
                    distances = np.linalg.norm(centers - point.x, axis=1)
                    labels.append(np.argmin(distances))

                labels = np.array(labels)

                # Compute sum of squared distances within each cluster
                ssq = 0
                for i, point in enumerate(_):
                    cluster_label = labels[i]
                    cluster_center = centers[cluster_label]
                    ssq += np.sum((point.x - cluster_center) ** 2)
                ssq_scores.append(ssq)
        return ssq_scores
    
    def _bss_score(self, cluster: Cluster, past_dps=None, future_dps=None):
        if cluster.implements_macro_clusters():
            centers = np.array(cluster._get_clusters_centers())
        else:
            centers = np.array(cluster._get_micro_clusters_centers())

        if centers.size == 0 or centers is None:
            return [0, 0]

        bss_scores = []
        if past_dps is not None and future_dps is not None:
            for _ in [past_dps, future_dps]:
                xs = np.array([p.x for p in _])
                overall_mean = np.mean(xs, axis=0)
                labels = []

                # Assign each data point to the nearest cluster center
                for point in _:
                    distances = np.linalg.norm(centers - point.x, axis=1)
                    labels.append(np.argmin(distances))

                labels = np.array(labels)

                # Compute between-cluster sum of squares (BSS)
                bss = 0
                for i, center in enumerate(centers):
                    cluster_size = np.sum(labels == i)
                    if cluster_size > 0:
                        bss += cluster_size * np.sum((center - overall_mean) ** 2)
                bss_scores.append(bss)

        return bss_scores

    def update(self, cluster: Cluster, past_dps=None, future_dps=None):
        if self.cluster_name is None:
            self.cluster_name = str(cluster)
        self.instances_seen += 1
        if self.instances_seen % self.update_interval == 0:
            self._update_measurements(cluster, past_dps=past_dps, future_dps=future_dps)

    def _update_measurements(self, cluster: Cluster, past_dps=None, future_dps=None):
        # update centers, weights, sizes, and radii
        try:
            if cluster.implements_macro_clusters():
                macro = cluster.get_clustering_result()
                if len(macro.get_centers()) > 0:
                    self.measurements["macros"].append(macro)
        except:
            pass

        if cluster.implements_micro_clusters():
            micro = cluster.get_micro_clustering_result()
            if len(micro.get_centers()) > 0:
                self.measurements["micros"].append(micro)

        if past_dps is not None and future_dps is not None:
            silhouette_scores = self._silhouette_score(cluster, past_dps=past_dps, future_dps=future_dps)
            ssq_scores = self._ssq_score(cluster, past_dps=past_dps, future_dps=future_dps)
            bss_scores = self._bss_score(cluster, past_dps=past_dps, future_dps=future_dps)

            if silhouette_scores:
                self.measurements["silhouette_past"].append(silhouette_scores[0])
                self.measurements["silhouette_future"].append(silhouette_scores[1])

            if ssq_scores:
                self.measurements["ssq_past"].append(ssq_scores[0])
                self.measurements["ssq_future"].append(ssq_scores[1])

            if bss_scores:
                self.measurements["bss_past"].append(bss_scores[0])
                self.measurements["bss_future"].append(bss_scores[1])
        # calculate silhouette score
        # TODO: delegate silhouette to moa
        # Check how it is done among different clusters

    def metrics_header(self):
        performance_names = ["macros", "micros", "silhouette_past", "silhouette_future", "ssq_past", "ssq_future", "bss_past", "bss_future"]
        return performance_names

    def metrics(self):
        # using the static list to keep the order of the metrics
        return [self.measurements[key] for key in self.metrics_header()]

    def get_measurements(self):
        return self.measurements

def online_evaluation_clustering(
        stream,
        clustream,
        max_instances=None,
        update_interval=1000,
    ):
    evaluator = ClusteringEvaluator(update_interval=update_interval)
    sliding_window = []
    stream.restart()

    start_wallclock_time, start_cpu_time = start_time_measuring()

    for i, instance in enumerate(stream):
        if max_instances is not None and i > max_instances:
            break
        # initializing phase
        if i <= update_interval:
            sliding_window.append(instance)
        elif i < 2 * update_interval:
            sliding_window.append(instance)
            clustream.train(sliding_window[-update_interval])
        else:
            sliding_window.append(instance)
            sliding_window.pop(0)
            clustream.train(sliding_window[-update_interval])
            past_dps = sliding_window[:-update_interval]
            future_dps = sliding_window[-update_interval:]
            # if evaluator.instances_seen % update_interval == 0:
            evaluator.update(clustream, past_dps, future_dps)

    elapsed_wallclock_time, elapsed_cpu_time = stop_time_measuring(start_wallclock_time, start_cpu_time)

    clustering_result = PrequentialClusteringResults(
        stream=stream,
        learner=clustream,
        wallclock=elapsed_wallclock_time,
        cpu_time=elapsed_cpu_time,
        max_instances=max_instances,
        macro_cluster=evaluator.measurements.pop('macros'),
        micro_cluster=evaluator.measurements.pop('micros'),
        other_metrics=evaluator.measurements,
    )

    return clustering_result

