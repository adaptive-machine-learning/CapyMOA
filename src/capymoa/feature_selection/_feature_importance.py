from __future__ import annotations

import inspect
from typing import Any, Optional

from capymoa.base import Classifier, MOAClassifier
from capymoa.stream import Schema


def _load_feature_importance_learners() -> tuple[Any, Any]:
    try:
        from moa.learners.featureanalysis import (
            FeatureImportanceHoeffdingTree,
            FeatureImportanceHoeffdingTreeEnsemble,
        )
    except Exception as exc:  # pragma: no cover - depends on moa.jar capabilities
        raise ImportError(
            "MOA feature-analysis learners are unavailable. Ensure your MOA jar "
            "contains `moa.learners.featureanalysis` classes."
        ) from exc
    return FeatureImportanceHoeffdingTree, FeatureImportanceHoeffdingTreeEnsemble


def _moa_learner(classifier: Any) -> Any:
    return (
        classifier.moa_learner if isinstance(classifier, MOAClassifier) else classifier
    )


def _coerce_base_learner(
    base_learner: Any,
    schema: Optional[Schema],
    random_seed: int,
) -> Any:
    if base_learner is None:
        return None

    if isinstance(base_learner, MOAClassifier):
        return base_learner

    if inspect.isclass(base_learner):
        if issubclass(base_learner, MOAClassifier):
            return base_learner(schema=schema, random_seed=random_seed)
        return base_learner()

    return base_learner


def _canonical_name(classifier: Any) -> str:
    return str(_moa_learner(classifier).getClass().getCanonicalName()).lower()


def _has_feature_importance(classifier: Any) -> bool:
    return hasattr(_moa_learner(classifier), "getFeatureImportances")


def _is_ensemble(classifier: Any) -> bool:
    name = _canonical_name(classifier)
    return any(
        token in name
        for token in (
            "adaptiverandomforest",
            "bag",
            "boost",
            "streamingrandompatches",
            "streaminggradientboostedtrees",
        )
    )


def _is_tree(classifier: Any) -> bool:
    name = _canonical_name(classifier)
    return any(token in name for token in ("trees.", "hoeffding"))


class FeatureImportanceClassifier(Classifier):
    """Base class for classifiers that expose feature-importance estimates."""

    def __init__(
        self,
        schema: Optional[Schema] = None,
        random_seed: int = 1,
        window_size: Optional[int] = None,
    ):
        super().__init__(schema=schema, random_seed=random_seed)

        if window_size is not None and window_size <= 0:
            raise ValueError("window_size must be a positive integer or None.")

        self.window_size = window_size
        self.instances_seen = 0
        self.feature_importances_per_window: Optional[list[dict[str, Any]]] = (
            [] if window_size is not None else None
        )

    def _on_train_complete(self) -> None:
        self.instances_seen += 1

        if (
            self.window_size is not None
            and self.feature_importances_per_window is not None
            and self.instances_seen % self.window_size == 0
        ):
            self.feature_importances_per_window.append(
                {
                    "instances_seen": self.instances_seen,
                    "importances": self.get_feature_importances(),
                }
            )

    def get_feature_importances(self, normalize: bool = True) -> list[float]:
        """Return the current feature importance scores."""
        raise NotImplementedError

    def get_top_k_features(self, k: int, normalize: bool = True) -> list[int]:
        importances = self.get_feature_importances(normalize=normalize)
        ranked_features = sorted(
            range(len(importances)),
            key=lambda feature_idx: importances[feature_idx],
            reverse=True,
        )
        return ranked_features[:k]

    def get_windowed_feature_importances(self) -> Optional[list[dict[str, Any]]]:
        return self.feature_importances_per_window


class MOAFeatureImportanceClassifier(FeatureImportanceClassifier, MOAClassifier):
    """Wrap a classifier with MOA's feature-importance learners.

    Accepted ``base_learner`` inputs:
    - CapyMOA ``MOAClassifier`` instance
    - CapyMOA ``MOAClassifier`` class
    - raw MOA learner instance
    - raw MOA learner class
    """

    def __init__(
        self,
        schema: Optional[Schema] = None,
        base_learner: Any = None,
        random_seed: int = 1,
        window_size: Optional[int] = None,
    ):
        FeatureImportanceClassifier.__init__(
            self,
            schema=schema,
            random_seed=random_seed,
            window_size=window_size,
        )
        base_learner = _coerce_base_learner(base_learner, schema, random_seed)
        moa_learner = self._build_moa_learner(base_learner)
        MOAClassifier.__init__(
            self,
            moa_learner=moa_learner,
            schema=schema,
            random_seed=random_seed,
        )

    @staticmethod
    def _build_moa_learner(base_learner: Any) -> Any:
        (
            feature_importance_tree,
            feature_importance_ensemble,
        ) = _load_feature_importance_learners()

        if base_learner is None:
            return feature_importance_tree()

        if _has_feature_importance(base_learner):
            return _moa_learner(base_learner)

        if _is_ensemble(base_learner):
            learner = feature_importance_ensemble()
            learner.ensembleLearnerOption.setCurrentObject(_moa_learner(base_learner))
            return learner

        if _is_tree(base_learner):
            learner = feature_importance_tree()
            learner.treeLearnerOption.setCurrentObject(_moa_learner(base_learner))
            return learner

        raise TypeError(
            "Feature importance is only supported for tree-based or ensemble MOA classifiers."
        )

    def train(self, instance: Any) -> None:
        MOAClassifier.train(self, instance)
        self._on_train_complete()

    def get_feature_importances(self, normalize: bool = True) -> list[float]:
        return list(self.moa_learner.getFeatureImportances(normalize))

__all__ = [
    "FeatureImportanceClassifier",
    "MOAFeatureImportanceClassifier",
]
