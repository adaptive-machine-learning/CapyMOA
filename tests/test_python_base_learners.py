"""Meta-learners take Python base learners through Java proxies.

Uses pure-Python learners only. MOA wrappers such as ``NaiveBayes`` or ``ADWIN``
are copied as Java objects and never reach a proxy.
"""

import math

import pytest
from moa.classifiers.meta import OzaBag as JOzaBag
from moa.classifiers.trees import LAST as JLAST

from capymoa.automl import BanditClassifier, SuccessiveHalvingClassifier
from capymoa.classifier import (
    CSMOTE,
    LAST,
    AdaptiveRandomForestClassifier,
    DynamicEnsembleMemberSelection,
    DynamicWeightedMajority,
    LeveragingBagging,
    OnlineAdwinBagging,
    OnlineBagging,
    OnlineSmoothBoost,
    OzaBoost,
    SGDClassifier,
    StreamingGradientBoostedTrees,
    StreamingRandomPatches,
)
from capymoa.core.moa._options import set_moa_options
from capymoa.core.moa.proxy import (
    ProxyJavaClassifier,
    ProxyJavaDriftDetector,
    ProxyJavaRegressor,
)
from capymoa.datasets import ElectricityTiny, FriedTiny
from capymoa.drift.detectors import ABCD, STUDD
from capymoa.evaluation import prequential_evaluation
from capymoa.regressor import (
    OnlineBaggingRegressor,
    SGDRegressor,
    StreamingGradientBoostedRegression,
)
from capymoa.ssl import SLEADE


@pytest.mark.parametrize(
    "java_learner, option, make_learner, proxy_type",
    [
        (
            JOzaBag,
            "baseLearner",
            lambda: SGDClassifier(ElectricityTiny().schema),
            ProxyJavaClassifier,
        ),
        (
            JOzaBag,
            "baseLearner",
            lambda: SGDRegressor(FriedTiny().schema),
            ProxyJavaRegressor,
        ),
        (JLAST, "changeDetectionMethodOption", ABCD, ProxyJavaDriftDetector),
    ],
    ids=["classifier", "regressor", "drift_detector"],
)
def test_option_wraps_python_learner(java_learner, option, make_learner, proxy_type):
    """A Python learner becomes a proxy that rebuilds it from its params."""
    moa_learner = java_learner()
    learner = make_learner()
    set_moa_options(moa_learner, options={option: learner})

    options = {str(o.getName()): o for o in moa_learner.getOptions().getOptionArray()}
    proxy = options[option].getPreMaterializedObject()
    assert isinstance(proxy, proxy_type)
    assert proxy._learner_spec["params"] == learner.get_params()


def sgd(schema):
    return SGDClassifier(schema)


CLASSIFIERS = {
    "OnlineBagging": lambda s: OnlineBagging(s, base_learner=sgd(s), ensemble_size=3),
    "LeveragingBagging": lambda s: LeveragingBagging(
        s, base_learner=sgd(s), ensemble_size=3
    ),
    "OnlineAdwinBagging": lambda s: OnlineAdwinBagging(
        s, base_learner=sgd(s), ensemble_size=3
    ),
    "OzaBoost": lambda s: OzaBoost(s, base_learner=sgd(s), boosting_iterations=3),
    "OnlineSmoothBoost": lambda s: OnlineSmoothBoost(
        s, base_learner=sgd(s), boosting_iterations=3
    ),
    "StreamingRandomPatches": lambda s: StreamingRandomPatches(
        s,
        base_learner=sgd(s),
        ensemble_size=3,
        drift_detection_method=ABCD(),
        warning_detection_method=ABCD(),
    ),
    "AdaptiveRandomForestClassifier": lambda s: AdaptiveRandomForestClassifier(
        s,
        ensemble_size=3,
        drift_detection_method=ABCD(),
        warning_detection_method=ABCD(),
    ),
    "DynamicEnsembleMemberSelection": lambda s: DynamicEnsembleMemberSelection(
        s,
        ensemble_size=3,
        drift_detection_method=ABCD(),
        warning_detection_method=ABCD(),
    ),
    "DynamicWeightedMajority": lambda s: DynamicWeightedMajority(
        s, base_learner=sgd(s)
    ),
    "CSMOTE": lambda s: CSMOTE(s, base_learner=sgd(s)),
    "LAST": lambda s: LAST(s, change_detector=ABCD()),
    "SLEADE": lambda s: SLEADE(
        s,
        ensemble_size=3,
        student_learner=sgd(s),
        drift_detection_method=ABCD(),
    ),
    "BanditClassifier": lambda s: BanditClassifier(
        s, base_classifiers=[sgd(s), SGDClassifier(s, loss="log_loss")]
    ),
    "SuccessiveHalvingClassifier": lambda s: SuccessiveHalvingClassifier(
        s, base_classifiers=[sgd(s), SGDClassifier(s, loss="log_loss")], budget=100
    ),
}


@pytest.mark.parametrize("make_learner", CLASSIFIERS.values(), ids=CLASSIFIERS.keys())
def test_classifier_with_python_learners(make_learner):
    stream = ElectricityTiny()
    learner = make_learner(stream.schema)
    results = prequential_evaluation(stream, learner, max_instances=300)
    # Majority class gets 58.7%.
    assert results["cumulative"]["accuracy"] > 65


def test_regressor_with_python_learner():
    stream = FriedTiny()
    learner = OnlineBaggingRegressor(
        stream.schema, base_learner=SGDRegressor(stream.schema), ensemble_size=3
    )
    results = prequential_evaluation(stream, learner, max_instances=300)
    assert math.isfinite(results["cumulative"]["rmse"])


def test_studd_with_python_learners():
    stream = ElectricityTiny()
    studd = STUDD(SGDClassifier(stream.schema), min_n_instances=100, detector=ABCD())
    for _ in range(300):
        instance = stream.next_instance()
        studd.add_element(instance, instance.y_index)
    assert studd.detector.idx == 201


@pytest.mark.parametrize(
    "make_learner",
    [
        lambda: StreamingGradientBoostedTrees(
            ElectricityTiny().schema, base_learner=SGDRegressor(FriedTiny().schema)
        ),
        lambda: StreamingGradientBoostedRegression(
            FriedTiny().schema, base_learner=SGDRegressor(FriedTiny().schema)
        ),
    ],
    ids=["SGBT", "SGBR"],
)
def test_sgbt_rejects_python_learner(make_learner):
    """SGBT copies its base learner with Java serialization, which proxies lack."""
    with pytest.raises(TypeError, match="Python learner"):
        make_learner()
