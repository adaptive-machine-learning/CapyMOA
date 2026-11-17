import inspect

from moa.classifiers.predictioninterval import AdaptivePredictionInterval as MOA_AdaPI

from capymoa.base import (
    MOAPredictionIntervalLearner,
    MOARegressor,
)
from capymoa.core.moa._cli import cli_str_regressor
from capymoa.regressor import AdaptiveRandomForestRegressor
from capymoa.stream import Schema


class AdaPI(MOAPredictionIntervalLearner):
    def __init__(
        self,
        schema: Schema | None = None,
        base_learner: str | MOARegressor | None = None,
        confidence_level: float = 0.95,
        limit: float = 0.1,
        random_seed: int | None = None,
    ) -> None:
        """Construct an AdaPI prediction interval learner.

        :param schema: Describes the stream's structure and types.
        :param base_learner: The base regressor to use. It can be a CLI string or a
            CapyMOA regressor. If not provided, an AdaptiveRandomForestRegressor is
            used.
        :param confidence_level: The confidence level of the prediction interval.
        :param limit: The threshold used to adapt the interval width.
        :param random_seed: Seed for all random choices. ``None`` is a fresh random seed each run.
        """
        mappings = {
            "base_learner": "-l",
            "confidence_level": "-c",
            "limit": "-t",
        }

        config_str = ""
        parameters = inspect.signature(self.__init__).parameters
        for key, cli_flag in mappings.items():
            if key not in parameters:
                continue
            this_parameter = parameters[key]
            set_value = locals()[key]
            if isinstance(set_value, bool):
                if set_value:
                    str_extension = cli_flag + " "
                else:
                    str_extension = ""
            else:
                if key == "base_learner":
                    if base_learner is None:
                        set_value = cli_str_regressor(
                            AdaptiveRandomForestRegressor(schema)
                        )
                    elif type(base_learner) is str:
                        set_value = base_learner
                    else:
                        set_value = cli_str_regressor(base_learner)

                str_extension = f"{cli_flag} {set_value} "
            config_str += str_extension

        self.moa_learner = MOA_AdaPI()

        self.moa_learner.getOptions().setViaCLIString(config_str)
        self.moa_learner.prepareForUse()
        self.moa_learner.resetLearning()

        super().__init__(
            schema=schema,
            random_seed=random_seed,
            moa_learner=self.moa_learner,
        )

    def __str__(self):
        # Overrides the default class name from MOA
        return "AdaptivePredictionInterval"
