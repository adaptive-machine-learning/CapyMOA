from __future__ import annotations

from capymoa.base import (
    MOAClassifier,
)
from capymoa.stream import Schema
from capymoa._utils import build_cli_str_from_mapping_and_locals

from moa.classifiers.functions import MajorityClass as _MOA_MajorityClass


class MajorityClass(MOAClassifier):
    """Majority class classifier.

    Always predicts the class that has been observed most frequently the in the training
    data.

    >>> from capymoa.datasets import ElectricityTiny
    >>> from capymoa.classifier import MajorityClass
    >>> from capymoa.evaluation import prequential_evaluation
    >>> stream = ElectricityTiny()
    >>> schema = stream.get_schema()
    >>> learner = MajorityClass(schema)
    >>> results = prequential_evaluation(stream, learner, max_instances=1000)
    >>> results["cumulative"].accuracy()
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
        super(MajorityClass, self).__init__(
            moa_learner=_MOA_MajorityClass,
            schema=schema,
            CLI=config_str,
        )
