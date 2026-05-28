# SPDX-License-Identifier: BSD-3-Clause
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class SparseConfig:
    """Global policy for automatic sparse Java instance creation."""

    enabled: bool = True
    sparsity_threshold: float = 0.9
    min_dimension: int = 100
    max_nonzero: int = 10000

    @classmethod
    def enable_auto_sparse(cls) -> None:
        cls.enabled = True

    @classmethod
    def disable_auto_sparse(cls) -> None:
        cls.enabled = False

    @classmethod
    def set_threshold(cls, threshold: float) -> None:
        if not 0.0 <= threshold <= 1.0:
            raise ValueError("sparsity threshold must be between 0 and 1")
        cls.sparsity_threshold = threshold


def _num_elements(x: Any) -> int:
    shape = getattr(x, "shape", None)
    if shape is not None:
        return int(np.prod(shape))
    return int(np.asarray(x).size)


def _num_nonzero(x: Any) -> int:
    nnz = getattr(x, "nnz", None)
    if nnz is not None:
        return int(nnz)
    return int(np.count_nonzero(np.asarray(x)))


def _as_1d_numeric_array(x: Any) -> np.ndarray:
    arr = np.asarray(x, dtype=float)
    if arr.ndim != 1:
        raise ValueError("feature vector must be one-dimensional")
    if arr.size == 0:
        raise ValueError("feature vector must not be empty")
    return arr


def calculate_sparsity(x: Any) -> float:
    """Return the fraction of zero-valued features in *x*."""

    total = _num_elements(x)
    if total == 0:
        raise ValueError("cannot calculate sparsity for an empty vector")
    return 1.0 - (_num_nonzero(x) / total)


def should_use_sparse(x: Any, force: bool | None = None) -> bool:
    """Decide whether *x* should be represented as a Java SparseInstance."""

    if force is not None:
        return bool(force)
    if not SparseConfig.enabled:
        return False

    total = _num_elements(x)
    nonzero = _num_nonzero(x)
    if total < SparseConfig.min_dimension:
        return False
    if nonzero > SparseConfig.max_nonzero:
        return False
    return calculate_sparsity(x) >= SparseConfig.sparsity_threshold


def _java_instance_classes():
    from jpype import JArray, JDouble, JInt

    from openmoa._prepare_jpype import _start_jpype

    _start_jpype()
    from com.yahoo.labs.samoa.instances import DenseInstance, SparseInstance

    return DenseInstance, SparseInstance, JArray, JDouble, JInt


def create_dense_java_instance(x: Any):
    """Create a MOA DenseInstance from a one-dimensional feature vector."""

    arr = _as_1d_numeric_array(x)
    DenseInstance, _, JArray, JDouble, _ = _java_instance_classes()
    values = np.empty(arr.size + 1, dtype=float)
    values[: arr.size] = arr
    values[-1] = np.nan
    return DenseInstance(1.0, JArray(JDouble)(values.tolist()))


def create_sparse_java_instance(x: Any):
    """Create a MOA SparseInstance from a one-dimensional feature vector."""

    arr = _as_1d_numeric_array(x)
    indices = np.flatnonzero(arr)
    if indices.size == 0:
        raise ValueError("sparse Java instances require at least one non-zero feature")

    _, SparseInstance, JArray, JDouble, JInt = _java_instance_classes()
    java_indices = indices.astype(int).tolist() + [int(arr.size)]
    java_values = arr[indices].astype(float).tolist() + [float("nan")]
    return SparseInstance(
        1.0,
        JArray(JDouble)(java_values),
        JArray(JInt)(java_indices),
        int(arr.size + 1),
    )


def create_java_instance(x: Any, force_sparse: bool | None = None):
    """Create a dense or sparse Java instance according to the sparse policy."""

    if should_use_sparse(x, force=force_sparse):
        return create_sparse_java_instance(x)
    return create_dense_java_instance(x)


def get_storage_info(x: Any) -> dict[str, float | int | str]:
    """Return memory and sparsity diagnostics for a feature vector."""

    total = _num_elements(x)
    if total == 0:
        raise ValueError("feature vector must not be empty")
    nonzero = _num_nonzero(x)
    dense_memory = total * 8
    sparse_memory = nonzero * (8 + 4)
    return {
        "num_features": total,
        "num_nonzero": nonzero,
        "sparsity": calculate_sparsity(x),
        "dense_memory_bytes": dense_memory,
        "sparse_memory_bytes": sparse_memory,
        "memory_saving_ratio": (dense_memory - sparse_memory) / dense_memory,
        "recommended": "sparse" if should_use_sparse(x) else "dense",
    }