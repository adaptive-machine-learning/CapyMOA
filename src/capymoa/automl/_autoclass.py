from capymoa.base import (
    Classifier,
    MOAClassifier,
)
from capymoa.stream import Schema
from capymoa._utils import build_cli_str_from_mapping_and_locals
from moa.classifiers.meta.AutoML import AutoClass as _MOA_AUTOCLASS
import os


class AutoClass(MOAClassifier):
    """AutoClass

    Reference:
    `Maroua Bahri, Nikolaos Georgantas.
    Autoclass: Automl for data stream classification.
    In BigData, IEEE, 2023. <https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=10386362>`_

    """

    def __init__(
        self,
        schema: Schema = None,
        random_seed: int = 0,
        configuration_json: str = "../../data/settings_autoclass.json",
        base_classifiers: list[Classifier] = [
            "lazy.kNN",
            "trees.HoeffdingTree",
            "trees.HoeffdingAdaptiveTree",
        ],
        number_active_classifiers: int = 1,
        weight_classifiers: bool = False,
    ):
        """AutoClass automl algorithm by Bahri and Georgantas.

        Note that configuration json file reading is delegated to the MOA object, thus in the configuration file
        the name of the learners should correspond to the MOA class full name.

        :param schema: The schema of the stream.
        :param random_seed: The random seed passed to the MOA learner.
        :param configuration: A json file with the configuration for learners
        :param base_classifiers: The learners that compose the ensemble
        :param number_active_classifiers: The number of active classifiers (used for voting)
        :param weight_classifiers: Uses online performance estimation to weight the classifiers
        """

        # Check if the json configuration file exists.
        if not os.path.exists(configuration_json):
            raise FileNotFoundError(
                f"The configuration json file was not found: {configuration_json}"
            )

        mapping = {
            # Configuration json file or dictionary
            "configuration_json": "-f",
            # How many instances before we re-evaluate the best classifier
            # "grace_period": "-g", not used currently
            # The classifiers the ensemble consists of
            "base_classifiers": "-b",
            # The number of active classifiers (used for voting)
            "number_active_classifiers": "-k",
            # Uses online performance estimation to weight the classifiers
            "weight_classifiers": "-p",
        }

        if all(isinstance(classifier, str) for classifier in base_classifiers):
            # Join the list of strings as 'x,y'
            base_classifiers = ",".join(base_classifiers)
        # Check if base_classifiers is a list of Classifier objects
        elif all(
            issubclass(classifier, MOAClassifier) for classifier in base_classifiers
        ):
            # Join the strings from the classifiers' class names
            base_classifiers = ",".join(
                str(classifier(schema).moa_learner.getClass().getName())
                for classifier in base_classifiers
            )
        else:
            raise ValueError(
                "base_classifiers must be either a list of strings or a list of Classifier objects"
            )

        config_str = build_cli_str_from_mapping_and_locals(mapping, locals())
        super(AutoClass, self).__init__(
            moa_learner=_MOA_AUTOCLASS,
            schema=schema,
            CLI=config_str,
            random_seed=random_seed,
        )
