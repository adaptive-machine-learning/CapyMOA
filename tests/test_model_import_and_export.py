import pytest
from capymoa.misc import save_model, load_model
from capymoa.classifier import AdaptiveRandomForestClassifier
from capymoa.datasets import ElectricityTiny
from capymoa.evaluation import prequential_evaluation
import os


def test_save_and_load_model_with_valid_model():
    # Arrange
    stream = ElectricityTiny()
    schema = stream.get_schema()
    learner = AdaptiveRandomForestClassifier(schema)
    prequential_evaluation(stream, learner, max_instances=1000)
    filename = 'test_model.pkl'

    # Act
    save_model(learner, filename)
    loaded_model = load_model(filename)

    # Assert
    assert isinstance(loaded_model, AdaptiveRandomForestClassifier)
    assert loaded_model != learner
    assert loaded_model.base_learner == learner.base_learner
    os.remove(filename)


def test_load_model_with_non_existent_file():
    # Arrange
    filename = 'non_existent_file.pkl'

    # Act & Assert
    with pytest.raises(FileNotFoundError):
        load_model(filename)

