import json
import tempfile

from capymoa.automl import BanditClassifier, EpsilonGreedy
from capymoa.stream.generator import SEA
from capymoa.base import _extract_moa_learner_CLI


def test_bandit_classifier_parameter_initialization():
    """Test that BanditClassifier passes parameters through the constructor."""

    # Create a config file with a HoeffdingTree(grace_period=50)
    config = {
        "algorithms": [
            {
                "algorithm": "HoeffdingTree",
                "parameters": [
                    {"parameter": "grace_period", "value": 50},
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

    # Instantiate BanditClassifier from config
    bc = BanditClassifier(
        schema=schema,
        config_file=config_path,
        policy=EpsilonGreedy(epsilon=0.1, burn_in=1),
    )

    # Retrieve the created model
    model = bc.active_models[0]

    # Check that the constructor parameter was applied (CLI should include "-g 50")
    cli_str = _extract_moa_learner_CLI(model)
    assert "-g 50" in cli_str, f"Expected grace_period=50, got CLI: {cli_str}"

    # Basic functional check
    instance = next(iter(stream))
    model.predict(instance)
    model.train(instance)
