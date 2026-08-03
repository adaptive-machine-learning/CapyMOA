import json
import tempfile

from capymoa.automl import SuccessiveHalvingClassifier
from capymoa.stream.generator import SEA
from capymoa._cli import cli_str_classifier


def test_successive_halving_parameter_initialization():
    """Test that SuccessiveHalvingClassifier passes parameters through the constructor."""

    # Create a config file with a HoeffdingTree(grace_period=30)
    config = {
        "algorithms": [
            {
                "algorithm": "HoeffdingTree",
                "parameters": [
                    {"parameter": "grace_period", "value": 30},
                ],
            }
        ]
    }

    with tempfile.NamedTemporaryFile("w", delete=False) as f:
        json.dump(config, f)
        config_path = f.name

    # Build a small stream and schema
    stream = SEA()
    schema = stream.get_schema()

    # Instantiate SuccessiveHalvingClassifier from config
    sh = SuccessiveHalvingClassifier(
        schema=schema,
        config_file=config_path,
        budget=10,
    )

    # Retrieve the created model
    model = sh.active_models[0]

    # Check that constructor parameters were applied ("-g 30" for grace_period)
    cli_str = cli_str_classifier(model)
    assert "-g 30" in cli_str, f"Expected grace_period=30, got CLI: {cli_str}"

    # Basic functional check
    instance = next(iter(stream))
    model.predict(instance)
    model.train(instance)
