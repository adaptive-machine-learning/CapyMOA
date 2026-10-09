"""The result type of :func:`~capymoa.ssl.evaluate_ssl`."""

from capymoa.classifier.evaluate import ClassifierResults


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
