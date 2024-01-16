from stream import stream_from_file
from evaluation import prequential_SSL_evaluation
from CPSSDS import CPSSDS
import pytest


@pytest.fixture
def stream():
    rbf_arff_file_path = "./data/RBFm_100k.arff"
    stream = stream_from_file(path_to_csv_or_arff=rbf_arff_file_path, class_index=-1)
    return stream


def eval(stream, cpssds):
    return prequential_SSL_evaluation(
        stream=stream,
        learner=cpssds,
        label_probability=0.5,
        window_size=10,
        optimise=False,
        max_instances=1000,
    )


def test_CPSSDS_NaiveBayes(stream):
    cpssds = CPSSDS("NaiveBayes", 100, schema=stream.schema)
    results_cl_100 = eval(stream, cpssds)
    assert results_cl_100["cumulative"].accuracy() == pytest.approx(31.7)


def test_CPSSDS_HoeffdingTree(stream):
    cpssds = CPSSDS("HoeffdingTree", 100, schema=stream.schema)
    results_cl_100 = eval(stream, cpssds)
    assert results_cl_100["cumulative"].accuracy() == pytest.approx(38.4)
