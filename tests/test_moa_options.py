import numpy as np
import pytest
from moa.classifiers.meta import OzaBag as JOzaBag
from moa.classifiers.trees import HoeffdingTree as JHoeffdingTree

import capymoa  # noqa: F401 (initializes JPype with MOA)
from capymoa.base import MOAClassifier
from capymoa.classifier import HoeffdingTree, OnlineBagging
from capymoa.core.moa._options import set_moa_options
from capymoa.datasets import ElectricityTiny
from capymoa.evaluation import prequential_evaluation


def test_unknown_option_name_raises():
    with pytest.raises(ValueError):
        set_moa_options(JHoeffdingTree(), options={"noSuchOption": 1})


def test_cli_and_options_are_mutually_exclusive():
    schema = ElectricityTiny().get_schema()
    with pytest.raises(ValueError):
        MOAClassifier(JHoeffdingTree, schema, CLI="-g 50", options={"gracePeriod": 50})


def test_options_match_cli_string():
    schema = ElectricityTiny().get_schema()
    from_cli = MOAClassifier(JHoeffdingTree, schema, CLI="-g 50")
    from_options = MOAClassifier(JHoeffdingTree, schema, options={"gracePeriod": 50})
    assert (
        from_options.moa_learner.getOptions().getAsCLIString()
        == from_cli.moa_learner.getOptions().getAsCLIString()
    )


def test_online_bagging_java_base_learner():
    """A raw Java base learner gives the same accuracy as the same CapyMOA one."""
    ensemble_size = 7
    grace_period = 33
    max_instances = 300

    stream = ElectricityTiny()
    schema = stream.get_schema()
    # Same config as the CapyMOA wrapper, which sets more than gracePeriod.
    capymoa_tree = HoeffdingTree(schema, grace_period=grace_period)
    java_tree = JHoeffdingTree()
    set_moa_options(
        java_tree, cli=str(capymoa_tree.moa_learner.getOptions().getAsCLIString())
    )
    java_learner = OnlineBagging(
        schema, base_learner=java_tree, ensemble_size=ensemble_size
    )
    java_results = prequential_evaluation(
        stream, java_learner, max_instances=max_instances
    )

    stream = ElectricityTiny()
    schema = stream.get_schema()
    capymoa_learner = OnlineBagging(
        schema,
        base_learner=HoeffdingTree(schema, grace_period=grace_period),
        ensemble_size=ensemble_size,
    )
    capymoa_results = prequential_evaluation(
        stream, capymoa_learner, max_instances=max_instances
    )

    assert java_results.cumulative.accuracy() == capymoa_results.cumulative.accuracy()


def test_wrapper_base_learner_not_mutated():
    """A CapyMOA wrapper used as base learner keeps its predictions."""
    stream = ElectricityTiny()
    schema = stream.get_schema()
    tree = HoeffdingTree(schema)
    for _ in range(200):
        tree.train(stream.next_instance())
    test_instances = [stream.next_instance() for _ in range(20)]
    before = [tree.predict_proba(i) for i in test_instances]

    OnlineBagging(schema, base_learner=tree, ensemble_size=3)
    after = [tree.predict_proba(i) for i in test_instances]

    for b, a in zip(before, after):
        np.testing.assert_allclose(b, a)


def test_bad_class_option_value_raises():
    """A value that is not a CLI string, wrapper or Java object raises TypeError."""
    with pytest.raises(TypeError, match="Cannot set MOA option"):
        set_moa_options(JOzaBag(), options={"baseLearner": object()})


def test_multichoice_option():
    """A multichoice option accepts a label name and stores it."""
    ht = JHoeffdingTree()
    set_moa_options(ht, options={"leafprediction": "MC"})
    assert str(ht.getOptions().getOption("leafprediction").getChosenLabel()) == "MC"


def test_online_bagging_minibatch_trains():
    """OnlineBagging with number_of_jobs trains and beats 50% accuracy."""
    schema = ElectricityTiny().get_schema()
    learner = OnlineBagging(schema, ensemble_size=3, number_of_jobs=2)
    assert learner.minibatch_size == 25
    results = prequential_evaluation(ElectricityTiny(), learner, max_instances=100)
    # Shut down MOA's thread pool, or its threads keep the JVM alive at exit.
    learner.moa_learner.trainingHasEnded()
    assert results.cumulative.accuracy() > 50

    learner = OnlineBagging(schema, ensemble_size=3, number_of_jobs=2, minibatch_size=5)
    results = prequential_evaluation(ElectricityTiny(), learner, max_instances=100)
    # Shut down MOA's thread pool, or its threads keep the JVM alive at exit.
    learner.moa_learner.trainingHasEnded()
    assert results.cumulative.accuracy() > 50
