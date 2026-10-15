"""The result type of :func:`~capymoa.ssl.evaluate_ssl`."""

from typing import NotRequired

from capymoa.classifier.evaluate import ClassifierCumulative, ClassifierResults


class SSLCumulative(ClassifierCumulative):
    """The metrics over the whole stream of a semi-supervised classifier."""

    #: Accuracy of the pseudo labels that the learner gave to unlabeled
    #: instances, from 0 to 1. Only set by the Java loop, if the learner
    #: makes pseudo labels.
    pseudo_label_accuracy: NotRequired[float]


class SSLResults(ClassifierResults):
    """Results of evaluating a semi-supervised classifier.

    See :func:`~capymoa.ssl.evaluate_ssl`.
    """

    #: Proportion of instances that had a label, from 0 to 1.
    label_probability: float
    #: Number of instances a label arrived late, 0 if labels are not delayed.
    delay_length: int
    #: Number of instances used to train before testing starts.
    initial_window_size: int
    #: Number of instances that never get a label. Instances whose label
    #: arrives late are not counted.
    unlabeled: int
    #: ``unlabeled`` divided by ``instances``.
    unlabeled_ratio: float
    #: The metrics over the whole stream (see
    #: :class:`~capymoa.ssl.evaluate.SSLCumulative`).
    cumulative: SSLCumulative
