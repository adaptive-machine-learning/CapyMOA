from __future__ import annotations

from capymoa.base import (
    MOAClassifier,
)
from capymoa.stream import Schema
from capymoa._utils import build_cli_str_from_mapping_and_locals

from moa.classifiers.functions import NoChange as _MOA_NoChange


class NoChange(MOAClassifier):
    """No change classifier.

    Always predicts the last class seen.

    >>> from capymoa.datasets import ElectricityTiny
    >>> from capymoa.classifier import NoChange
    >>> from capymoa.evaluation import prequential_evaluation
    >>> stream = ElectricityTiny()
    >>> schema = stream.get_schema()
    >>> learner = NoChange(schema)
    >>> results = prequential_evaluation(stream, learner, max_instances=1000)
    >>> results["cumulative"].accuracy()
    85.9
    """

    def __init__(
        self,
        schema: Schema | None = None,
    ):
        """NoChange class classifier.

        :param schema: The schema of the stream.
        """

        mapping = {}

        config_str = build_cli_str_from_mapping_and_locals(mapping, locals())
        super(NoChange, self).__init__(
            moa_learner=_MOA_NoChange,
            schema=schema,
            CLI=config_str,
        )
