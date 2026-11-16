import time
from itertools import islice

from capymoa.base import Clusterer
from capymoa.cluster.evaluate import ClusteringEvaluator, ClusterResults
from capymoa.stream import Stream


def evaluate_cluster(
    stream: Stream,
    clusterer: Clusterer,
    max_instances: int | None = None,
    update_interval: int = 1000,
    restart_stream: bool = True,
) -> ClusterResults:
    """Train a clusterer on a stream and record its clusters.

    The clusterer is trained on each instance in turn. Every ``update_interval``
    instances, the clusters are saved. The last instances are saved too, if they
    do not end on an update.

    >>> from capymoa.cluster import Clustream
    >>> from capymoa.stream.generator import RandomRBFGeneratorDrift
    >>> stream = RandomRBFGeneratorDrift(number_of_attributes=2)
    >>> clusterer = Clustream(schema=stream.get_schema())
    >>> results = evaluate_cluster(
    ...     stream, clusterer, max_instances=500, update_interval=100
    ... )
    >>> results["instances"], len(results["micro"]) > 0
    (500, True)

    :param stream: The stream to train on. Restarted if ``restart_stream``.
    :param clusterer: The clusterer to train. It is trained in place.
    :param max_instances: Number of instances to train on. If ``None``, go on
        until the stream ends.
    :param update_interval: Save the clusters every this many instances.
    :param restart_stream: If ``False``, continue from the current position in the
        stream.
    :return: The results, with the cluster snapshots.
    """
    if update_interval < 1:
        raise ValueError("update_interval must be at least 1.")
    if restart_stream:
        stream.restart()

    evaluator = ClusteringEvaluator(update_interval=update_interval)
    start_wallclock, start_cpu = time.time(), time.process_time()
    instances = 0
    for instance in islice(stream, max_instances):
        clusterer.train(instance)
        evaluator.update(clusterer)
        instances += 1
    if instances % update_interval:
        evaluator._update_measurements(clusterer)
    wallclock = time.time() - start_wallclock
    cpu_time = time.process_time() - start_cpu

    measurements = evaluator.get_measurements()
    return ClusterResults(
        learner=str(clusterer),
        stream=str(stream),
        instances=instances,
        wallclock=wallclock,
        cpu_time=cpu_time,
        macro=measurements["macro"],
        micro=measurements["micro"],
    )  # type: ignore[typeddict-item]
