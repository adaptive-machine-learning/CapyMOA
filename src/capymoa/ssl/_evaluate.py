from collections import deque
from typing import Any

import numpy as np
from tqdm import tqdm
from typing_extensions import override

from capymoa.base import Classifier, ClassifierSSL
from capymoa.classifier._evaluate import _classifier_results
from capymoa.classifier.evaluate import (
    ClassificationEvaluator,
    ClassificationWindowedEvaluator,
)
from capymoa.evaluation._loop import (
    _prequential_loop,
    _prequential_loop_fast,
    _Run,
    _use_java_loop,
)
from capymoa.ssl._results import SSLResults
from capymoa.stream import Stream


class _SSLRun(_Run):
    """Test-then-train, keeping only some labels."""

    def __init__(
        self,
        learner: Classifier | ClassifierSSL,
        cumulative,
        windowed,
        *,
        delay_length: int,
        label_probability: float,
        random_seed: int,
        store_y: bool,
        store_predictions: bool,
    ):
        super().__init__(
            learner,
            cumulative,
            windowed,
            store_y=store_y,
            store_predictions=store_predictions,
        )
        self.delay_length = delay_length
        self.label_probability = label_probability
        mt19937 = np.random.MT19937()
        mt19937._legacy_seeding(random_seed)
        self._rand = np.random.Generator(mt19937)
        self._seen = 0
        # Instances whose label is delayed: each entry is the index at which the
        # instance reappears as labeled, paired with the instance itself.
        self._delayed: deque[tuple[int, Any]] = deque()
        # Instances that never get a label.
        self.unlabeled = 0

    @override
    def test_then_train(self, batch) -> tuple[list[Any], list[Any]]:
        learner = self.learner
        y_true, y_pred = [], []
        for instance in batch:
            # Deliver any labels whose delay has elapsed, before this instance
            # is used, so the learner has everything available up to this point.
            while self._delayed and self._delayed[0][0] <= self._seen:
                learner.train(self._delayed.popleft()[1])

            y_pred.append(learner.predict(instance))
            y_true.append(instance.y_index)

            if self._rand.random(dtype=np.float64) >= self.label_probability:
                # Do not label the instance. Otherwise, just ignore it.
                if isinstance(learner, ClassifierSSL):
                    learner.train_on_unlabeled(instance)
                self.unlabeled += 1
            elif self.delay_length > 0:
                # The label exists but arrives late: the instance is presented
                # unlabeled now and reappears as labeled after ``delay_length``
                # instances.
                if isinstance(learner, ClassifierSSL):
                    learner.train_on_unlabeled(instance)
                self._delayed.append((self._seen + self.delay_length, instance))
            else:
                learner.train(instance)
            self._seen += 1
        return y_true, y_pred


def _unlabeled(unlabeled: int, instances: int) -> dict[str, Any]:
    """The ``unlabeled`` and ``unlabeled_ratio`` keys of a result."""
    ratio = unlabeled / instances if instances else float("nan")
    return {"unlabeled": unlabeled, "unlabeled_ratio": ratio}


def evaluate_ssl(
    stream: Stream,
    learner: ClassifierSSL | Classifier,
    max_instances: int | None = None,
    window_size: int | None = 1000,
    initial_window_size: int = 0,
    delay_length: int = 0,
    label_probability: float = 0.01,
    random_seed: int = 1,
    store_predictions: bool = False,
    store_y: bool = False,
    optimise: bool = True,
    restart_stream: bool = True,
    progress_bar: bool | tqdm = False,
) -> SSLResults:
    """Test-then-train a semi-supervised classifier on a stream.

    Only some instances keep their label.

    >>> from capymoa.classifier import NaiveBayes
    >>> from capymoa.datasets import ElectricityTiny
    >>> from capymoa.ssl import evaluate_ssl
    >>> stream = ElectricityTiny()
    >>> results = evaluate_ssl(
    ...     stream, NaiveBayes(stream.get_schema()), max_instances=1000
    ... )
    >>> results["label_probability"]
    0.01

    :param stream: The stream to evaluate on. Restarted if ``restart_stream``.
    :param learner: The learner to evaluate. A :class:`~capymoa.base.ClassifierSSL` also trains
        on unlabeled instances. Any other classifier trains on labeled instances
        only.
    :param max_instances: Number of instances to evaluate. If ``None``, go on
        until the stream ends.
    :param window_size: Number of instances in a window of ``windowed``. If
        ``None``, there are no windowed results.
    :param initial_window_size: Number of instances to train on (with labels)
        before testing. Only the Java loop supports it.
    :param delay_length: If above zero, a label arrives this many instances after
        its instance, which first appears unlabeled.
    :param label_probability: The proportion of instances with a label, from 0 to
        1.
    :param random_seed: The seed that decides which instances have a label.
    :param store_predictions: Keep the predictions in ``y_pred``.
    :param store_y: Keep the ground truth in ``y_true``.
    :param optimise: Use the Java loop in MOA if the learner allows it. Needs a
        ``window_size``.
    :param restart_stream: If ``False``, continue from the current position in the
        stream.
    :param progress_bar: Enable, disable, or give a progress bar. The Java loop
        has no progress bar.
    :return: The results.
    """
    if restart_stream:
        stream.restart()
    schema = stream.get_schema()
    if not schema.is_classification():
        raise ValueError("The learning task is not classification")
    if delay_length < 0:
        raise ValueError("delay_length must be zero or positive.")
    extra = {
        "label_probability": label_probability,
        "delay_length": delay_length,
        "initial_window_size": initial_window_size,
    }

    if _use_java_loop(stream, learner, optimise=optimise, window_size=window_size):
        run = _Run(
            learner,
            ClassificationEvaluator(schema=schema),
            ClassificationWindowedEvaluator(schema=schema, window_size=window_size),
            store_y=store_y,
            store_predictions=store_predictions,
        )
        out = _prequential_loop_fast(
            stream,
            run,
            max_instances=max_instances,
            ssl=(initial_window_size, delay_length, label_probability, random_seed),
        )
        base = _classifier_results(
            str(learner), stream, out, run.cumulative, run.windowed
        )
        unlabeled = int(out.other.get("num_unlabeled_instances", 0))
        return SSLResults(**base, **extra, **_unlabeled(unlabeled, out.instances))  # type: ignore[typeddict-item]

    # `initial_window_size` is only implemented in MOA, for the Java loop.
    if initial_window_size != 0:
        raise ValueError(
            "Initial window size must be 0 for this function as the feature is not implemented yet."
        )
    name = str(learner)
    windowed = None
    if window_size is not None:
        windowed = ClassificationWindowedEvaluator(
            schema=schema, window_size=window_size
        )
    run = _SSLRun(
        learner,
        ClassificationEvaluator(schema=schema),
        windowed,
        delay_length=delay_length,
        label_probability=label_probability,
        random_seed=random_seed,
        store_y=store_y,
        store_predictions=store_predictions,
    )
    out = _prequential_loop(
        stream,
        {name: run},
        max_instances=max_instances,
        progress_bar=progress_bar,
        progress_prefix="SSL Eval",
    )[name]
    base = _classifier_results(name, stream, out, run.cumulative, run.windowed)
    return SSLResults(**base, **extra, **_unlabeled(run.unlabeled, out.instances))  # type: ignore[typeddict-item]
