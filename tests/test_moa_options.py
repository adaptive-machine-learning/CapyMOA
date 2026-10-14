import pytest
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


def test_online_bagging_minibatch_builds():
    schema = ElectricityTiny().get_schema()
    OnlineBagging(schema, ensemble_size=3, number_of_jobs=2)
