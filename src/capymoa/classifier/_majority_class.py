from __future__ import annotations

from moa.classifiers.functions import MajorityClass as _MOA_MajorityClass

from capymoa._utils import build_cli_str_from_mapping_and_locals
from capymoa.base import (
    MOAClassifier,
)
from capymoa.stream import Schema


class MajorityClass(MOAClassifier):
    """Majority class classifier.

    Always predicts the class that has been observed most frequently the in the training
    data.

    >>> from capymoa.datasets import ElectricityTiny
    >>> from capymoa.classifier import MajorityClass
    >>> from capymoa.classifier import evaluate_classifier
    >>> stream = ElectricityTiny()
    >>> schema = stream.get_schema()
    >>> learner = MajorityClass(schema)
    >>> results = evaluate_classifier(stream, learner, max_instances=1000)
    >>> results["accuracy"]
    50.2
    """

    def __init__(
        self,
        schema: Schema | None = None,
    ):
        """Majority class classifier.

        :param schema: The schema of the stream.
        """

        mapping = {}

        config_str = build_cli_str_from_mapping_and_locals(mapping, locals())
        super().__init__(
            moa_learner=_MOA_MajorityClass,
            schema=schema,
            CLI=config_str,
        )
