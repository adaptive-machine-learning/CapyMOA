"""Evaluation procedures and evaluators for CapyMOA learners.

This module provides prequential evaluation functions and evaluator classes for
classification, regression, prediction interval, anomaly detection, and
clustering tasks.
"""

from .evaluation import (
    prequential_evaluation,
    prequential_evaluation_multiple_learners,
    prequential_ssl_evaluation,
    prequential_evaluation_anomaly,
    ClassificationEvaluator,
    ClassificationWindowedEvaluator,
    RegressionWindowedEvaluator,
    RegressionEvaluator,
    PredictionIntervalEvaluator,
    PredictionIntervalWindowedEvaluator,
    AnomalyDetectionEvaluator,
    ClusteringEvaluator,
)
from . import results

__all__ = [
    "prequential_evaluation",
    "prequential_ssl_evaluation",
    "prequential_evaluation_multiple_learners",
    "prequential_evaluation_anomaly",
    "ClassificationEvaluator",
    "ClassificationWindowedEvaluator",
    "RegressionWindowedEvaluator",
    "RegressionEvaluator",
    "PredictionIntervalEvaluator",
    "PredictionIntervalWindowedEvaluator",
    "AnomalyDetectionEvaluator",
    "ClusteringEvaluator",
    "results",
]
