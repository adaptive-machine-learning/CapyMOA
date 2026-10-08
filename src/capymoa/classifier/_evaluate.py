import math
import re
from collections.abc import Mapping

import numpy as np
from tqdm import tqdm

from capymoa.base import Classifier
from capymoa.classifier._results import ClassifierResults, PerClass
from capymoa.classifier.evaluate import (
    ClassificationEvaluator,
    ClassificationWindowedEvaluator,
)
from capymoa.evaluation._loop import (
    _LoopOutput,
    _prequential_loop,
    _prequential_loop_fast,
    _require_mapping,
    _require_single,
    _Run,
    _run_info,
    _use_java_loop,
)
from capymoa.stream import Schema, Stream

_METRICS = [
    "accuracy",
    "kappa",
    "kappa_t",
    "kappa_m",
    "f1_score",
    "precision",
    "recall",
]
_PER_CLASS_METRICS = ["precision", "recall", "f1_score"]
_PER_CLASS = re.compile(r"^(precision|recall|f1_score)_(\d+)$")


def _per_class(metrics: Mapping[str, float], schema: Schema) -> PerClass:
    labels = list(schema.get_label_values())
    columns = {name: np.full(len(labels), np.nan) for name in _PER_CLASS_METRICS}
    for key, value in metrics.items():
        match = _PER_CLASS.match(key)
        if match and int(match.group(2)) < len(labels):
            columns[match.group(1)][int(match.group(2))] = value
    return PerClass(label=np.array(labels), **columns)  # type: ignore[typeddict-item]


def _classifier_results(
    name: str,
    stream: Stream | str,
    out: _LoopOutput,
    cumulative: ClassificationEvaluator,
    windowed: ClassificationWindowedEvaluator | None,
) -> ClassifierResults:
    metrics = cumulative.metrics_dict()
    roc_auc = float(metrics["roc_auc"])
    results = ClassifierResults(
        **_run_info(name, stream, out, windowed, [*_METRICS, "roc_auc"]),
        **{key: float(metrics[key]) for key in _METRICS},
        per_class=_per_class(metrics, cumulative.schema),
    )  # type: ignore[typeddict-item]
    if not math.isnan(roc_auc):
        results["roc_auc"] = roc_auc
    return results


def evaluate_classifiers(
    stream: Stream,
    learners: Mapping[str, Classifier],
    max_instances: int | None = None,
    window_size: int | None = 1000,
    store_predictions: bool = False,
    store_y: bool = False,
    restart_stream: bool = True,
    progress_bar: bool | tqdm = False,
    batch_size: int = 1,
) -> dict[str, ClassifierResults]:
    """Test-then-train many classifiers on one pass over a stream.

    The learners get the same instances in turn, so the stream is read once.
    This is useful when reading the stream is slow. It does not use the Java
    loop. The training and testing of the learners is interleaved, so
    ``wallclock`` and ``cpu_time`` are for the whole pass and are the same for
    all the learners. Do not use them to compare the speed of learners.

    >>> from capymoa.classifier import HoeffdingTree, NaiveBayes, evaluate_classifiers
    >>> from capymoa.datasets import ElectricityTiny
    >>> stream = ElectricityTiny()
    >>> learners = {
    ...     "ht": HoeffdingTree(stream.get_schema()),
    ...     "nb": NaiveBayes(stream.get_schema()),
    ... }
    >>> results = evaluate_classifiers(stream, learners, max_instances=1000)
    >>> print(f"{results['ht']['accuracy']:.1f} {results['nb']['accuracy']:.1f}")
    84.4 84.8

    :param stream: The stream to evaluate on. Restarted if ``restart_stream``.
    :param max_instances: Number of instances to evaluate. If ``None``, go on
        until the stream ends.
    :param window_size: Number of instances in a window of ``windowed``. If
        ``None``, there are no windowed results.
    :param store_predictions: Keep the predictions in ``y_pred``.
    :param store_y: Keep the ground truth in ``y_true``.
    :param restart_stream: If ``False``, continue from the current position in the
        stream.
    :param progress_bar: Enable, disable, or give a progress bar.
    :param learners: The learners to evaluate, by name. The name is the ``learner``
        key of the result.
    :param batch_size: Instances per mini-batch, for batch learners.
    :return: The results by name.
    """
    learners = _require_mapping(learners, "evaluate_classifier")
    if restart_stream:
        stream.restart()
    schema = stream.get_schema()
    if not schema.is_classification():
        raise ValueError("The stream is not a classification stream.")

    runs = {
        n: _Run(
            one,
            ClassificationEvaluator(schema=schema),
            None
            if window_size is None
            else ClassificationWindowedEvaluator(
                schema=schema, window_size=window_size
            ),
            store_y=store_y,
            store_predictions=store_predictions,
        )
        for n, one in learners.items()
    }
    outs = _prequential_loop(
        stream,
        runs,
        max_instances=max_instances,
        progress_bar=progress_bar,
        batch_size=batch_size,
    )
    return {
        n: _classifier_results(n, stream, out, runs[n].cumulative, runs[n].windowed)
        for n, out in outs.items()
    }


def evaluate_classifier(
    stream: Stream,
    learner: Classifier,
    max_instances: int | None = None,
    window_size: int | None = 1000,
    store_predictions: bool = False,
    store_y: bool = False,
    optimise: bool = True,
    restart_stream: bool = True,
    progress_bar: bool | tqdm = False,
    batch_size: int = 1,
) -> ClassifierResults:
    """Test-then-train a classifier on a stream (prequential evaluation).

    Each instance is first used to test the learner, then to train it.

    >>> from capymoa.classifier import HoeffdingTree, evaluate_classifier
    >>> from capymoa.datasets import ElectricityTiny
    >>> stream = ElectricityTiny()
    >>> results = evaluate_classifier(
    ...     stream, HoeffdingTree(stream.get_schema()), max_instances=1000
    ... )
    >>> print(f"{results['accuracy']:.1f}")
    84.4

    To compare learners on one pass over the stream use
    :func:`evaluate_classifiers`.

    :param stream: The stream to evaluate on. Restarted if ``restart_stream``.
    :param max_instances: Number of instances to evaluate. If ``None``, go on
        until the stream ends.
    :param window_size: Number of instances in a window of ``windowed``. If
        ``None``, there are no windowed results.
    :param store_predictions: Keep the predictions in ``y_pred``.
    :param store_y: Keep the ground truth in ``y_true``.
    :param restart_stream: If ``False``, continue from the current position in the
        stream.
    :param progress_bar: Enable, disable, or give a progress bar.
    :param learner: The learner to evaluate.
    :param optimise: Use the Java loop in MOA if the learner allows it. Needs a
        ``window_size``. The Java loop has no progress bar.
    :param batch_size: Instances per mini-batch, for batch learners.
    :return: The results.
    """
    _require_single(learner, "evaluate_classifiers")
    if restart_stream:
        stream.restart()
    schema = stream.get_schema()
    if not schema.is_classification():
        raise ValueError("The stream is not a classification stream.")
    name = str(learner)
    if _use_java_loop(
        stream,
        learner,
        optimise=optimise,
        window_size=window_size,
        batch_size=batch_size,
    ):
        run = _Run(
            learner,
            ClassificationEvaluator(schema=schema),
            ClassificationWindowedEvaluator(schema=schema, window_size=window_size),
            store_y=store_y,
            store_predictions=store_predictions,
        )
        out = _prequential_loop_fast(stream, run, max_instances=max_instances)
        return _classifier_results(name, stream, out, run.cumulative, run.windowed)
    return evaluate_classifiers(
        stream,
        {name: learner},
        max_instances=max_instances,
        window_size=window_size,
        store_predictions=store_predictions,
        store_y=store_y,
        restart_stream=False,
        progress_bar=progress_bar,
        batch_size=batch_size,
    )[name]
