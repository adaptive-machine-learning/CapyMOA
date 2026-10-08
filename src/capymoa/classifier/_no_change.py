from __future__ import annotations

from moa.classifiers.functions import NoChange as _MOA_NoChange

from capymoa._utils import build_cli_str_from_mapping_and_locals
from capymoa.base import (
    MOAClassifier,
)
from capymoa.stream import Schema


class NoChange(MOAClassifier):
    """No change classifier.

    Always predicts the last class seen.

    >>> from capymoa.datasets import ElectricityTiny
    >>> from capymoa.classifier import NoChange
    >>> from capymoa.classifier import evaluate_classifier
    >>> stream = ElectricityTiny()
    >>> schema = stream.get_schema()
    >>> learner = NoChange(schema)
    >>> results = evaluate_classifier(stream, learner, max_instances=1000)
    >>> results["accuracy"]
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
        super().__init__(
            moa_learner=_MOA_NoChange,
            schema=schema,
            CLI=config_str,
        )
