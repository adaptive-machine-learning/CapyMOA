import pytest
from capymoa.misc import save_model, load_model
from capymoa.classifier import AdaptiveRandomForestClassifier
from capymoa.datasets import ElectricityTiny
from capymoa.evaluation import prequential_evaluation
import os


def test_save_and_load_model():
   
    stream = ElectricityTiny()
    schema = stream.get_schema()
    learner = AdaptiveRandomForestClassifier(schema)
    prequential_evaluation(stream, learner, max_instances=1000)
    filename = 'test_model.pkl'

    save_model(learner, filename)
    loaded_model = load_model(filename)

    assert isinstance(loaded_model, AdaptiveRandomForestClassifier)
    assert loaded_model != learner
    assert loaded_model.base_learner == learner.base_learner
    os.remove(filename)


def test_load_model_missing_file():
    filename = 'non_existent_file.pkl'

    with pytest.raises(FileNotFoundError):
        load_model(filename)

