import inspect

from capymoa.base import (
    MOAPredictionIntervalLearner,
    _extract_moa_learner_CLI,
)

from capymoa.regressor import AdaptiveRandomForestRegressor

from moa.classifiers.predictioninterval import MVEPredictionInterval as MOA_MVE


class MVE(MOAPredictionIntervalLearner):
    def __init__(
        self,
        schema=None,
        CLI=None,
        random_seed=1,
        base_learner=None,
        confidence_level=0.95,
    ):
        mappings = {"base_learner": "-l", "confidence_level": "-c"}

        config_str = ""
        parameters = inspect.signature(self.__init__).parameters
        for key in mappings:
            if key not in parameters:
                continue
            this_parameter = parameters[key]
            set_value = locals()[key]
            if isinstance(set_value, bool):
                if set_value:
                    str_extension = mappings[key] + " "
                else:
                    str_extension = ""
            else:
                if key == "base_learner":
                    if base_learner is None:
                        set_value = _extract_moa_learner_CLI(
                            AdaptiveRandomForestRegressor(schema)
                        )
                    elif type(base_learner) is str:
                        set_value = base_learner
                    else:
                        set_value = _extract_moa_learner_CLI(base_learner)

                str_extension = f"{mappings[key]} {set_value} "
            config_str += str_extension

        self.moa_learner = MOA_MVE()

        if CLI is None:
            self.moa_learner.getOptions().setViaCLIString(config_str)
            self.moa_learner.prepareForUse()
            self.moa_learner.resetLearning()

        super().__init__(
            schema=schema,
            CLI=CLI,
            random_seed=random_seed,
            moa_learner=self.moa_learner,
        )

    def __str__(self):
        # Overrides the default class name from MOA
        return "MVEPredictionInterval"
