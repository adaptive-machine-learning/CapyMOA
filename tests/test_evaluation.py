from capymoa.stream.generator import SEA
from capymoa.classifier import NaiveBayes, HoeffdingTree
from capymoa.evaluation import prequential_evaluation, \
    prequential_evaluation_multiple_learners, prequential_ssl_evaluation
import pytest


def test_prequential_evaluation():
    """The stream should be restarted every time we run the evaluation, so the 11th instance should be the same, also
        the accuracy of models from the same learner (but different models) should be the same
    """
    stream = SEA(function=1)
    model1 = NaiveBayes(schema=stream.get_schema())
    model2 = NaiveBayes(schema=stream.get_schema())

    results_1st_run = prequential_evaluation(stream=stream, learner=model1, max_instances=10)
    eleventh_instance_1st_run = results_1st_run['stream'].next_instance().x
    results_2nd_run = prequential_evaluation(stream=stream, learner=model2, max_instances=10)
    eleventh_instance_2nd_run = results_2nd_run['stream'].next_instance().x

    assert eleventh_instance_1st_run == pytest.approx(eleventh_instance_2nd_run)

    assert results_1st_run['cumulative'].accuracy() == pytest.approx(
        results_2nd_run['cumulative'].accuracy(), abs=0.001
    ), f"Prequential evaluation same synthetic stream: Expected accuracy of " \
       f"{results_1st_run['cumulative'].accuracy():0.3f} got {results_2nd_run['cumulative'].accuracy(): 0.3f}"


def test_prequential_evaluation_multiple_learners():
    """The stream should be restarted every time we run the evaluation, so the 11th instance should be the same, also
        the accuracy of models from the same learner (but different models) should be the same
    """
    stream = SEA(function=1)
    model11 = NaiveBayes(schema=stream.get_schema())
    model12 = HoeffdingTree(schema=stream.get_schema())
    model21 = NaiveBayes(schema=stream.get_schema())
    model22 = HoeffdingTree(schema=stream.get_schema())

    results_1st_run = prequential_evaluation_multiple_learners(stream=stream,
                                                               learners={'model11': model11, 'model12': model12},
                                                               max_instances=100)
    hundredth_first_instance_1st_run = results_1st_run['stream'].next_instance().x
    results_2nd_run = prequential_evaluation_multiple_learners(stream=stream,
                                                               learners={'model21': model21, 'model22': model22},
                                                               max_instances=100)
    hundredth_first_instance_2nd_run = results_2nd_run['stream'].next_instance().x

    assert hundredth_first_instance_1st_run == pytest.approx(hundredth_first_instance_2nd_run)

    assert results_1st_run['model11']['cumulative'].accuracy() == pytest.approx(
        results_2nd_run['model21']['cumulative'].accuracy(), abs=0.001
    ), f"Prequential evaluation multiple learners same synthetic stream: Expected accuracy of " \
       f"{results_1st_run['model11']['cumulative'].accuracy():0.3f} got " \
       f"{results_2nd_run['model21']['cumulative'].accuracy(): 0.3f}"

    assert results_1st_run['model12']['cumulative'].accuracy() == pytest.approx(
        results_2nd_run['model22']['cumulative'].accuracy(), abs=0.001
    ), f"Prequential evaluation multiple learners same synthetic stream: Expected accuracy of " \
       f"{results_1st_run['model12']['cumulative'].accuracy():0.3f} got " \
       f"{results_2nd_run['model22']['cumulative'].accuracy(): 0.3f}"


def test_prequential_ssl_evaluation():
    """The stream should be restarted every time we run the evaluation, so the 11th instance should be the same, also
        the accuracy of models from the same learner (but different models) should be the same
    """
    stream = SEA(function=1)
    model1 = NaiveBayes(schema=stream.get_schema())
    model2 = NaiveBayes(schema=stream.get_schema())

    results_1st_run = prequential_ssl_evaluation(stream=stream, learner=model1, max_instances=10)
    eleventh_instance_1st_run = results_1st_run['stream'].next_instance().x
    results_2nd_run = prequential_ssl_evaluation(stream=stream, learner=model2, max_instances=10)
    eleventh_instance_2nd_run = results_2nd_run['stream'].next_instance().x

    assert eleventh_instance_1st_run == pytest.approx(eleventh_instance_2nd_run)

    assert results_1st_run['cumulative'].accuracy() == pytest.approx(
        results_2nd_run['cumulative'].accuracy(), abs=0.001
    ), f"Prequential_ssl_evaluation same synthetic stream: Expected accuracy of " \
       f"{results_1st_run['cumulative'].accuracy():0.3f} got {results_2nd_run['cumulative'].accuracy(): 0.3f}"
