from capymoa.classifier import AdaptiveRandomForestClassifier
from capymoa.datasets import ElectricityTiny
from capymoa.misc import legacy_save_model, legacy_load_model, load_model
from tempfile import TemporaryDirectory
import pytest


def test_legacy_save_load_model():
    """Tests the legacy save and load model functions.

    Ensures using the new `load_model` function with a legacy model file raises
    an exception with a note explaining the error.
    """

    # TODO: This should be removed when `legacy_save_model` and `legacy_load_model`
    # are removed.
    stream = ElectricityTiny()
    learner = AdaptiveRandomForestClassifier(schema=stream.get_schema())
    with TemporaryDirectory() as tmpdir:
        filename = tmpdir + "/model.pkl"
        legacy_save_model(learner, filename)

        with pytest.raises(RuntimeError):
            with open(filename, "rb") as fd:
                load_model(fd)

        legacy_load_model(filename)
