from __future__ import annotations

import warnings

import numpy as np

from capymoa.base._seed import resolve_seed
from capymoa.core import Instance, LabeledInstance
from capymoa.stream import Schema
from capymoa.stream.preprocessing import Transformer

from ._feature_importance import (
    FeatureImportanceClassifier,
    MOAFeatureImportanceClassifier,
)


def _feature_names(schema: Schema) -> list[str]:
    """Return the names of the features, in the order of ``instance.x``."""
    header = schema.get_moa_header()
    return [
        str(header.attribute(i).name())
        for i in range(header.numAttributes())
        if i != header.classIndex()
    ]


class TopKFeatureSelector(Transformer):
    """Keep the ``k`` most important features, chosen after a warm-up.

    The selector trains a feature-importance classifier on the first
    ``warmup`` instances passed to :meth:`train`. It then keeps the ``k``
    features with the highest importance, in their original order, and drops
    the rest for the remainder of the stream. The selection does not change
    after that.

    Build the learner after the warm-up, from :meth:`get_schema`, so that it
    expects the selected features:

    >>> from capymoa.classifier import HoeffdingTree
    >>> from capymoa.datasets import ElectricityTiny
    >>> from capymoa.feature import TopKFeatureSelector
    >>> from capymoa.stream.preprocessing import ClassifierPipeline
    >>> stream = ElectricityTiny()
    >>> selector = TopKFeatureSelector(stream.get_schema(), k=3, warmup=500, random_seed=1)
    >>> for _ in range(500):
    ...     selector.train(stream.next_instance())
    >>> selector.selected_features
    ['period', 'nswprice', 'nswdemand']
    >>> pipeline = (
    ...     ClassifierPipeline()
    ...     .add_transformer(selector)
    ...     .add_classifier(HoeffdingTree(selector.get_schema()))
    ... )

    Before the warm-up ends, :meth:`get_schema` returns ``None`` and
    :meth:`transform_instance` raises, because the output is not known yet.

    Importance comes from a :class:`~capymoa.feature.FeatureImportanceClassifier`,
    an adaptive random forest by default. A single Hoeffding tree
    scores no feature until its first split, which can take longer than a
    warm-up.

    Changing the selection later in the stream would need downstream learners
    to accept a new feature set, so it is not supported yet.
    """

    def __init__(
        self,
        schema: Schema,
        k: int,
        warmup: int = 1000,
        scorer: FeatureImportanceClassifier | None = None,
        random_seed: int | None = None,
    ) -> None:
        """Construct a selector.

        :param schema: The schema of instances entering the selector.
        :param k: The number of features to keep.
        :param warmup: The number of training instances to score features on
            before selecting.
        :param scorer: A feature-importance classifier for ``schema``. If
            ``None``, an adaptive random forest is used.
        :param random_seed: Seed for the default scorer. Ignored when
            ``scorer`` is given.
        :raises ValueError: If the schema is not for classification, or ``k``
            or ``warmup`` is out of range.
        """
        if not schema.is_classification():
            raise ValueError(
                "TopKFeatureSelector needs a classification schema: its feature "
                "importances come from a classifier."
            )
        n_features = schema.get_num_attributes()
        if not 1 <= k <= n_features:
            raise ValueError(f"k must be between 1 and {n_features}, got {k}.")
        if warmup < 1:
            raise ValueError(f"warmup must be at least 1, got {warmup}.")

        self.schema = schema
        self.k = k
        self.warmup = warmup
        if scorer is None:
            from capymoa.classifier import AdaptiveRandomForestClassifier

            scorer = MOAFeatureImportanceClassifier(
                schema=schema,
                base_learner=AdaptiveRandomForestClassifier,
                random_seed=resolve_seed(random_seed),
            )
        self.scorer = scorer
        self.instances_seen = 0
        """The number of training instances scored so far."""
        self._selected: list[int] | None = None
        self._output_schema: Schema | None = None

    @property
    def is_selected(self) -> bool:
        """Whether the warm-up is over and the features are chosen."""
        return self._selected is not None

    @property
    def selected_features(self) -> list[str] | None:
        """The names of the kept features, or ``None`` before the warm-up ends."""
        if self._selected is None:
            return None
        names = _feature_names(self.schema)
        return [names[i] for i in self._selected]

    def train(self, instance: Instance) -> None:
        """Score features on a training instance, and select once warm-up ends.

        Does nothing once the features are selected.
        """
        if self._selected is not None:
            return
        self.scorer.train(instance)
        self.instances_seen += 1
        if self.instances_seen >= self.warmup:
            self._select()

    def _select(self) -> None:
        # A tree that has not split yet reports NaN for every feature, and
        # sorting NaNs gives an arbitrary order, so rank them as 0. A stable
        # sort breaks ties by the original feature order.
        scores = np.nan_to_num(
            np.asarray(self.scorer.get_feature_importances(), dtype=float), nan=0.0
        )
        ranked = np.argsort(-scores, kind="stable")
        self._selected = sorted(int(i) for i in ranked[: self.k])

        n_scored = int((scores > 0).sum())
        if n_scored < self.k:
            warnings.warn(
                f"Only {n_scored} of {len(scores)} features had a non-zero "
                f"importance after {self.instances_seen} instances, so "
                f"{self.k - n_scored} of the {self.k} kept were chosen by their "
                "position. Try a longer warmup.",
                stacklevel=3,
            )

        names = _feature_names(self.schema)
        nominal = self.schema.get_nominal_attributes() or {}
        kept = [names[i] for i in self._selected]
        target = str(self.schema.get_moa_header().classAttribute().name())
        categories = {name: nominal[name] for name in kept if name in nominal}
        categories[target] = list(self.schema.get_label_values())
        self._output_schema = Schema.from_custom(
            features=[*kept, target],
            target=target,
            categories=categories,
            name=f"{self.schema.dataset_name}-top{self.k}",
        )

    def transform_instance(self, instance: Instance) -> Instance:
        """Return the instance with only the selected features.

        :raises RuntimeError: If the warm-up has not ended.
        """
        if self._selected is None or self._output_schema is None:
            raise RuntimeError(
                "TopKFeatureSelector has not selected its features yet: call "
                f"train() on {self.warmup} instances first "
                f"({self.instances_seen} so far)."
            )
        x = np.asarray(instance.x)[self._selected]
        if isinstance(instance, LabeledInstance):
            return LabeledInstance.from_array(self._output_schema, x, instance.y_index)
        return Instance.from_array(self._output_schema, x)

    def get_schema(self) -> Schema | None:  # type: ignore[override]
        """Return the schema of the selected features, or ``None`` before then."""
        return self._output_schema

    def get_input_schema(self) -> Schema:
        """Return the schema of instances entering the selector."""
        return self.schema

    def restart(self) -> None:
        """Forget the selection and start a new warm-up.

        The scorer keeps what it has learnt, so a new warm-up continues from it.
        """
        self._selected = None
        self._output_schema = None
        self.instances_seen = 0

    def __str__(self) -> str:
        return f"TopKFeatureSelector(k={self.k}, warmup={self.warmup})"


__all__ = ["TopKFeatureSelector"]
