import math
import re
from collections.abc import Mapping
from typing import overload

import numpy as np
from tqdm import tqdm

from capymoa.base import Classifier
from capymoa.classifier.evaluate import (
    ClassificationEvaluator,
    ClassificationWindowedEvaluator,
    ClassifierResults,
    ClassifierWindows,
    PerClass,
)
from capymoa.evaluation._loop import (
    _LoopOutput,
    _prequential_loop,
    _prequential_loop_fast,
    _results_body,
    _Run,
    _use_java_loop,
)
from capymoa.stream import Schema, Stream

_PER_CLASS_METRICS = [key for key in PerClass.__annotations__ if key != "label"]
_PER_CLASS = re.compile(rf"^({'|'.join(_PER_CLASS_METRICS)})_(\d+)$")


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
    body = _results_body(name, stream, out, cumulative, windowed, ClassifierWindows)
    if math.isnan(body["roc_auc"]):
        del body["roc_auc"]
    body["per_class"] = _per_class(cumulative.metrics_dict(), cumulative.schema)
    return ClassifierResults(**body)  # type: ignore[typeddict-item]


@overload
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
) -> ClassifierResults: ...
@overload
def evaluate_classifier(
    stream: Stream,
    learner: Mapping[str, Classifier],
    max_instances: int | None = None,
    window_size: int | None = 1000,
    store_predictions: bool = False,
    store_y: bool = False,
    optimise: bool = True,
    restart_stream: bool = True,
    progress_bar: bool | tqdm = False,
    batch_size: int = 1,
) -> dict[str, ClassifierResults]: ...
def evaluate_classifier(
    stream: Stream,
    learner: Classifier | Mapping[str, Classifier],
    max_instances: int | None = None,
    window_size: int | None = 1000,
    store_predictions: bool = False,
    store_y: bool = False,
    optimise: bool = True,
    restart_stream: bool = True,
    progress_bar: bool | tqdm = False,
    batch_size: int = 1,
) -> ClassifierResults | dict[str, ClassifierResults]:
    """Test-then-train a classifier, or many on one pass over a stream.

    Each instance is first used to test the learner, then to train it.

    >>> from capymoa.classifier import HoeffdingTree, evaluate_classifier
    >>> from capymoa.datasets import ElectricityTiny
    >>> stream = ElectricityTiny()
    >>> results = evaluate_classifier(
    ...     stream, HoeffdingTree(stream.get_schema()), max_instances=1000
    ... )
    >>> print(f"{results['accuracy']:.1f}")
    84.4

    Give a mapping of names to learners to get a dict of results by name. The
    learners get the same instances in turn, so the stream is read once. This
    is useful when reading the stream is slow. A mapping does not use the Java
    loop. The learners are interleaved, so ``wallclock`` and ``cpu_time`` are
    for the whole pass and are the same for all the learners. Do not use them
    to compare the speed of learners.

    >>> from capymoa.classifier import NaiveBayes
    >>> learners = {
    ...     "ht": HoeffdingTree(stream.get_schema()),
    ...     "nb": NaiveBayes(stream.get_schema()),
    ... }
    >>> results = evaluate_classifier(stream, learners, max_instances=1000)
    >>> print(f"{results['ht']['accuracy']:.1f} {results['nb']['accuracy']:.1f}")
    84.4 84.8

    :param stream: The stream to evaluate on. Restarted if ``restart_stream``.
    :param learner: The learner to evaluate, or a mapping of names to learners.
        The name is the ``learner`` key of each result.
    :param max_instances: Number of instances to evaluate. If ``None``, go on
        until the stream ends.
    :param window_size: Number of instances in a window of ``windowed``. If
        ``None``, there are no windowed results.
    :param store_predictions: Keep the predictions in ``y_pred``.
    :param store_y: Keep the ground truth in ``y_true``.
    :param optimise: Use the Java loop in MOA if the learner allows it. Needs a
        ``window_size``. The Java loop has no progress bar. Only used for one
        learner.
    :param restart_stream: If ``False``, continue from the current position in the
        stream.
    :param progress_bar: Enable, disable, or give a progress bar.
    :param batch_size: Instances per mini-batch, for batch learners.
    :return: The results, or a dict of results by name for a mapping.
    """
    many = isinstance(learner, Mapping)
    if many and not learner:
        raise ValueError("No learners to evaluate.")
    if restart_stream:
        stream.restart()
    schema = stream.get_schema()
    if not schema.is_classification():
        raise ValueError("The stream is not a classification stream.")

    if not many and _use_java_loop(
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
        return _classifier_results(
            str(learner), stream, out, run.cumulative, run.windowed
        )

    learners = dict(learner) if many else {str(learner): learner}
    runs = {}
    for name, each in learners.items():
        windowed = None
        if window_size is not None:
            windowed = ClassificationWindowedEvaluator(
                schema=schema, window_size=window_size
            )
        runs[name] = _Run(
            each,
            ClassificationEvaluator(schema=schema),
            windowed,
            store_y=store_y,
            store_predictions=store_predictions,
        )
    outs = _prequential_loop(
        stream,
        runs,
        max_instances=max_instances,
        progress_bar=progress_bar,
        batch_size=batch_size,
    )
    results = {
        n: _classifier_results(n, stream, out, runs[n].cumulative, runs[n].windowed)
        for n, out in outs.items()
    }
    return results if many else next(iter(results.values()))
