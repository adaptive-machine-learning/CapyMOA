from __future__ import annotations

from typing import Sequence

import matplotlib.pyplot as plt
import pandas as pd


def _normalize_feature_names(
    feature_names: Sequence[str] | str | None, n_features: int
) -> list[str]:
    if feature_names is None:
        return [f"f{i}" for i in range(n_features)]
    if isinstance(feature_names, str):
        names = [feature_names]
    else:
        names = [str(name) for name in feature_names]

    if len(names) != n_features:
        raise ValueError(
            f"feature_names length ({len(names)}) does not match importances length ({n_features})."
        )
    return names


def plot_feature_importance(
    importances: Sequence[float],
    feature_names: Sequence[str] | str | None = None,
    *,
    top_k: int | None = None,
    ax=None,
    title: str = "Feature importances",
):
    """Plot feature importances as a bar chart."""
    values = list(importances)
    names = _normalize_feature_names(feature_names, len(values))

    series = pd.Series(values, index=names, name="importance").sort_values(
        ascending=False
    )
    if top_k is not None:
        if top_k <= 0:
            raise ValueError("top_k must be greater than zero.")
        series = series.head(top_k)

    if ax is None:
        _, ax = plt.subplots(figsize=(10, 4))

    series.plot(kind="bar", ax=ax)
    ax.set_title(title)
    ax.set_xlabel("Feature")
    ax.set_ylabel("Importance")
    ax.figure.tight_layout()
    return ax


def plot_windowed_feature_importance(
    windowed_importances: list[dict],
    feature_names: Sequence[str] | str | None = None,
    *,
    top_k: int | None = None,
    ax=None,
    title: str = "Windowed feature importances",
):
    """Plot windowed feature importances over time."""
    if len(windowed_importances) == 0:
        raise ValueError("windowed_importances is empty.")
    if top_k is not None and top_k <= 0:
        raise ValueError("top_k must be greater than zero.")

    n_features = len(windowed_importances[0]["importances"])
    names = _normalize_feature_names(feature_names, n_features)

    window_df = pd.DataFrame(
        [entry["importances"] for entry in windowed_importances],
        columns=names,
    )
    window_df["instances_seen"] = [
        entry["instances_seen"] for entry in windowed_importances
    ]

    if top_k is None:
        top_features = names
    else:
        top_features = (
            window_df[names]
            .mean()
            .sort_values(ascending=False)
            .head(top_k)
            .index.tolist()
        )

    if ax is None:
        _, ax = plt.subplots(figsize=(10, 4))

    window_df.plot(x="instances_seen", y=top_features, ax=ax)
    ax.set_title(title)
    ax.set_xlabel("Instances seen")
    ax.set_ylabel("Importance")
    ax.figure.tight_layout()
    return ax
