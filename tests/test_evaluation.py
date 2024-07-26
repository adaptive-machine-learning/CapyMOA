from capymoa.evaluation.evaluation import prequential_evaluation_anomaly
from capymoa.stream.generator import SEA
from capymoa.classifier import NaiveBayes, HoeffdingTree
from capymoa.evaluation import (prequential_evaluation,
                                prequential_evaluation_multiple_learners,
                                prequential_ssl_evaluation
                                )
from capymoa.datasets import ElectricityTiny
import pytest
from capymoa.datasets import Electricity
from capymoa.anomaly import (
    HalfSpaceTrees,
)


def test_prequential_evaluation():
    """The stream should be restarted every time we run the evaluation, so the 11th instance should be the same, also
        the accuracy of models from the same learner (but different models) should be the same
    """
    stream = SEA(function=1)
    model1 = NaiveBayes(schema=stream.get_schema())
    model2 = NaiveBayes(schema=stream.get_schema())

    results_1st_run = prequential_evaluation(stream=stream, learner=model1, max_instances=10)
    eleventh_instance_1st_run = results_1st_run.stream.next_instance().x
    results_2nd_run = prequential_evaluation(stream=stream, learner=model2, max_instances=10)
    eleventh_instance_2nd_run = results_2nd_run.stream.next_instance().x

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
    # print(results_1st_run['model11'].stream)
    hundredth_first_instance_1st_run = results_1st_run['model11'].stream.next_instance().x
    results_2nd_run = prequential_evaluation_multiple_learners(stream=stream,
                                                               learners={'model21': model21, 'model22': model22},
                                                               max_instances=100)
    hundredth_first_instance_2nd_run = results_2nd_run['model21'].stream.next_instance().x

    assert hundredth_first_instance_1st_run == pytest.approx(hundredth_first_instance_2nd_run)

    assert results_1st_run['model11'].cumulative.accuracy() == pytest.approx(
        results_2nd_run['model21'].cumulative.accuracy(), abs=0.001
    ), f"Prequential evaluation multiple learners same synthetic stream: Expected accuracy of " \
       f"{results_1st_run['model11'].cumulative.accuracy():0.3f} got " \
       f"{results_2nd_run['model21'].cumulative.accuracy(): 0.3f}"

    assert results_1st_run['model12'].cumulative.accuracy() == pytest.approx(
        results_2nd_run['model22'].cumulative.accuracy(), abs=0.001
    ), f"Prequential evaluation multiple learners same synthetic stream: Expected accuracy of " \
       f"{results_1st_run['model12'].cumulative.accuracy():0.3f} got " \
       f"{results_2nd_run['model22'].cumulative.accuracy(): 0.3f}"


def test_prequential_ssl_evaluation():
    """The stream should be restarted every time we run the evaluation, so the 11th instance should be the same, also
        the accuracy of models from the same learner (but different models) should be the same
    """
    stream = SEA(function=1)
    model1 = NaiveBayes(schema=stream.get_schema())
    model2 = NaiveBayes(schema=stream.get_schema())

    results_1st_run = prequential_ssl_evaluation(stream=stream, learner=model1, max_instances=10)
    eleventh_instance_1st_run = results_1st_run.stream.next_instance().x
    results_2nd_run = prequential_ssl_evaluation(stream=stream, learner=model2, max_instances=10)
    eleventh_instance_2nd_run = results_2nd_run.stream.next_instance().x

    assert eleventh_instance_1st_run == pytest.approx(eleventh_instance_2nd_run)

    assert results_1st_run.cumulative.accuracy() == pytest.approx(
        results_2nd_run.cumulative.accuracy(), abs=0.001
    ), f"Prequential_ssl_evaluation same synthetic stream: Expected accuracy of " \
       f"{results_1st_run.cumulative.accuracy():0.3f} got {results_2nd_run.cumulative.accuracy(): 0.3f}"


def _test_accessibility(obj, function_names):
        errors = []
        for func_name in function_names:
            try:
                # Check if the function is directly accessible
                if not hasattr(obj, func_name):
                    raise AttributeError(f"Function {func_name} is not directly accessible.")

                # Attempt to call the function if it's callable
                func = getattr(obj, func_name)
                if callable(func):
                    func()
                else:
                    raise AttributeError(f"{func_name} is not callable.")

                # Check if the function is accessible via __getitem__
                if obj[func_name] is None:  # func_name in obj.metrics_header():
                    raise KeyError(f"{func_name} is not accessible via __getitem__.")

            except Exception as e:
                errors.append((func_name, str(e)))

        return errors


def test_evaluation_api():
    """Test whether the API is functioning as expected, the access to result objects and so on.
    """

    # Define the list of function names that should be accessible through results_ht
    prequential_results_function_names = [
        'wallclock',
        'cpu_time',
        'max_instances',
        'ground_truth_y',
        'predictions'
    ]

    stream = ElectricityTiny()
    ht = HoeffdingTree(schema=stream.get_schema(), grace_period=50)

    results_ht = prequential_evaluation(stream=stream, learner=ht, window_size=50, optimise=True,
                                        store_predictions=True, store_y=True)

    # Test accessibility of PrequentialResults attributes through PrequentialResults object. This is relevant
    # as the function tests access via __getitem__ (i.e. []), like results['wallclock']
    results_ht_errors = _test_accessibility(results_ht, prequential_results_function_names)
    if results_ht_errors:
        print("Errors accessing PrequentialResults attributes through PrequentialResults object: ")
        for func_name, error in results_ht_errors:
            print(f"{func_name}: {error}")
    else:
        print("PrequentialResults attributes are accessible.")

    # Test accessibility of cumulative functions through prequential results object
    cumulative_errors = _test_accessibility(results_ht, results_ht.cumulative.metrics_header())
    if cumulative_errors:
        print("Errors accessing cumulative metrics through prequential results object: ")
        for func_name, error in cumulative_errors:
            print(f"{func_name}: {error}")
    else:
        print("Cumulative metrics are accessible through PrequentialResults object.")

    assert results_ht_errors == [], "Issues with access to PrequentialResults"
    assert cumulative_errors == [], "Issues with access to cumulative"


def test_prequential_evaluation_anomaly():
    """The stream should be restarted every time we run the evaluation, so the 11th instance should be the same, also
        the AUC of models from the same learner (but different models) should be the same
    """
    stream = Electricity()
    model1 = HalfSpaceTrees(schema=stream.get_schema())
    model2 = HalfSpaceTrees(schema=stream.get_schema())

    results_1st_run = prequential_evaluation_anomaly(stream=stream, learner=model1, window_size=1000, optimise=True)
    results_2nd_run = prequential_evaluation_anomaly(stream=stream, learner=model2, window_size=1000, optimise=False)

    assert results_1st_run['windowed'].auc() == pytest.approx(
        results_2nd_run['windowed'].auc(), abs=0.001
    ), f"prequential_evaluation_anomaly same synthetic stream: Expected AUC of " \
       f"{results_1st_run['windowed'].auc():0.3f} got {results_2nd_run['windowed'].auc(): 0.3f}"