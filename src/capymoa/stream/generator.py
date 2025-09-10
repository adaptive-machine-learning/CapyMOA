"""Generate artificial data streams."""

import copy

from capymoa.stream import MOAStream
from moa.streams.generators import RandomTreeGenerator as MOA_RandomTreeGenerator
from moa.streams.generators import SEAGenerator as MOA_SEAGenerator
from moa.streams.generators import HyperplaneGenerator as MOA_HyperplaneGenerator
from moa.streams.generators import (
    HyperplaneGeneratorForRegression as MOA_HyperplaneGeneratorForRegression,
)
from moa.streams.generators import RandomRBFGenerator as MOA_RandomRBFGenerator
from moa.streams.generators import (
    RandomRBFGeneratorDrift as MOA_RandomRBFGeneratorDrift,
)
from moa.streams.generators import AgrawalGenerator as MOA_AgrawalGenerator
from moa.streams.generators import LEDGenerator as MOA_LEDGenerator
from moa.streams.generators import LEDGeneratorDrift as MOA_LEDGeneratorDrift
from moa.streams.generators import WaveformGenerator as MOA_WaveformGenerator
from moa.streams.generators import WaveformGeneratorDrift as MOA_WaveformGeneratorDrift
from moa.streams.generators import STAGGERGenerator as MOA_STAGGERGenerator
from moa.streams.generators import SineGenerator as MOA_SineGenerator
from capymoa._utils import build_cli_str_from_mapping_and_locals


class RandomTreeGenerator(MOAStream):
    """Stream generator for a stream based on a randomly generated tree.

    >>> from capymoa.stream.generator import RandomTreeGenerator
    ...
    >>> stream = RandomTreeGenerator()
    >>> stream.next_instance()
    LabeledInstance(
        Schema(generators.RandomTreeGenerator ),
        x=[0.    3.    2.    3.    4.    0.036 0.659 0.711 0.153 0.16 ],
        y_index=0,
        y_label='class1'
    )
    >>> stream.next_instance().x
    array([4.        , 2.        , 2.        , 1.        , 4.        ,
           0.39717434, 0.34751803, 0.29405703, 0.50648363, 0.11596709])
    """

    def __init__(
        self,
        instance_random_seed: int = 1,
        tree_random_seed: int = 1,
        num_classes: int = 2,
        num_nominals: int = 5,
        num_numerics: int = 5,
        num_vals_per_nominal: int = 5,
        max_tree_depth: int = 5,
        first_leaf_level: int = 3,
        leaf_fraction: float = 0.15,
    ):
        """Construct a random tree generator.

        :param instance_random_seed: Seed for random generation of instances.
        :param tree_random_seed: Seed for random generation of tree.
        :param num_classes: The number of classes to generate.
        :param num_nominals: The number of nominal attributes to generate.
        :param num_numerics: The number of numeric attributes to generate.
        :param num_vals_per_nominal: The number of values to generate per nominal attribute.
        :param max_tree_depth: The maximum depth of the tree concept.
        :param first_leaf_level: The first level of the tree above ``max_tree_depth`` that can have leaves
        :param leaf_fraction: The fraction of leaves per level from first leaf level onwards.
        """
        self.__init_args_kwargs__ = copy.copy(
            locals()
        )  # save init args for recreation. not a deep copy to avoid unnecessary use of memory

        self.moa_stream = MOA_RandomTreeGenerator()

        self.instance_random_seed = instance_random_seed
        self.tree_random_seed = tree_random_seed
        self.num_classes = num_classes
        self.num_nominals = num_nominals
        self.num_numerics = num_numerics
        self.num_vals_per_nominal = num_vals_per_nominal
        self.max_tree_depth = max_tree_depth
        self.first_leaf_level = first_leaf_level
        self.leaf_fraction = leaf_fraction

        self.CLI = f"-i {instance_random_seed} -r {self.tree_random_seed} \
            -c {self.num_classes} -o {self.num_nominals} -u {self.num_numerics} -v {self.num_vals_per_nominal} \
            -d {max_tree_depth} -l {first_leaf_level} -f {leaf_fraction}"

        # self.moa_stream.getHeader()

        super().__init__(CLI=self.CLI, moa_stream=self.moa_stream)

    def __str__(self):
        attributes = [
            (
                f"instance_random_seed={self.instance_random_seed}"
                if self.instance_random_seed != 1
                else None
            ),
            (
                f"tree_random_seed={self.tree_random_seed}"
                if self.tree_random_seed != 1
                else None
            ),
            f"num_classes={self.num_classes}" if self.num_classes != 2 else None,
            f"num_nominals={self.num_nominals}" if self.num_nominals != 5 else None,
            f"num_numerics={self.num_numerics}" if self.num_numerics != 5 else None,
            (
                f"num_vals_per_nominal={self.num_vals_per_nominal}"
                if self.num_vals_per_nominal != 5
                else None
            ),
            (
                f"max_tree_depth={self.max_tree_depth}"
                if self.max_tree_depth != 5
                else None
            ),
            (
                f"first_leaf_level={self.first_leaf_level}"
                if self.first_leaf_level != 3
                else None
            ),
            (
                f"leaf_fraction={self.leaf_fraction}"
                if self.leaf_fraction != 0.15
                else None
            ),
        ]

        non_default_attributes = [attr for attr in attributes if attr is not None]
        return f"RTG({', '.join(non_default_attributes)})"


class SEA(MOAStream):
    """Generates SEA concepts functions.

    >>> from capymoa.stream.generator import SEA
    ...
    >>> stream = SEA()
    >>> stream.next_instance()
    LabeledInstance(
        Schema(generators.SEAGenerator ),
        x=[7.309 4.101 2.077],
        y_index=1,
        y_label='groupB'
    )
    >>> stream.next_instance().x
    array([6.58867239, 7.10739628, 1.52736201])

    Street, W. N., & Kim, Y. (2001). A streaming ensemble algorithm (SEA) for
    large-scale classification. :doi:`doi:10.1145/502512.502568<10.1145/502512.502568>`
    """

    def __init__(
        self,
        instance_random_seed: int = 1,
        function: int = 1,
        balance_classes: bool = False,
        noise_percentage: int = 10,
    ):
        """Construct a SEA datastream generator.

        :param instance_random_seed: Seed for random generation of instances, defaults to 1
        :param function: Classification function used, as defined in the original paper, defaults to 1
        :param balance_classes: Balance the number of instances of each class, defaults to False
        :param noise_percentage: Percentage of noise to add to the data, defaults to 10
        """
        self.__init_args_kwargs__ = copy.copy(
            locals()
        )  # save init args for recreation. not a deep copy to avoid unnecessary use of memory

        self.moa_stream = MOA_SEAGenerator()

        self.instance_random_seed = instance_random_seed
        self.function = function
        self.balance_classes = balance_classes
        self.noise_percentage = noise_percentage

        self.CLI = f"-i {instance_random_seed} -f {self.function} \
            {'-b' if self.balance_classes else ''} -p {self.noise_percentage}"

        super().__init__(CLI=self.CLI, moa_stream=self.moa_stream)

    def __str__(self):
        attributes = [
            (
                f"instance_random_seed={self.instance_random_seed}"
                if self.instance_random_seed != 1
                else None
            ),
            f"function={self.function}",
            f"balance_classes={self.balance_classes}" if self.balance_classes else None,
            (
                f"noise_percentage={self.noise_percentage}"
                if self.noise_percentage != 10
                else None
            ),
        ]
        non_default_attributes = [attr for attr in attributes if attr is not None]
        return f"SEA({', '.join(non_default_attributes)})"


class HyperPlaneClassification(MOAStream):
    """Generates HyperPlane concepts functions.

    >>> from capymoa.stream.generator import HyperPlaneClassification
    ...
    >>> stream = HyperPlaneClassification()
    >>> stream.next_instance()
    LabeledInstance(
        Schema(generators.HyperplaneGenerator ),
        x=[0.397 0.348 0.294 0.506 0.116 0.771 0.66  0.157 0.378 0.14 ],
        y_index=0,
        y_label='class1'
    )
    >>> stream.next_instance().x
    array([0.00485253, 0.85225356, 0.02341807, 0.70500995, 0.27502995,
           0.0753878 , 0.61059154, 0.95493077, 0.2740691 , 0.19020221])
    """

    def __init__(
        self,
        instance_random_seed: int = 1,
        number_of_classes: int = 2,
        number_of_attributes: int = 10,
        number_of_drifting_attributes: int = 2,
        magnitude_of_change: float = 0.0,
        noise_percentage: int = 5,
        sigma_percentage: int = 10,
    ):
        """Construct a HyperPlane Classification datastream generator.

        :param instance_random_seed: Seed for random generation of instances, defaults to 1
        :param number_of_classes: The number of classes of the generated instances, defaults to 2
        :param number_of_attributes: The number of attributes of the generated instances, defaults to 10
        :param number_of_drifting_attributes: The number of drifting attributes, defaults to 2
        :param magnitude_of_change: Magnitude of change in the generated instances, defaults to 0.0
        :param noise_percentage: Percentage of noise to add to the data, defaults to 10
        :param sigma_percentage: Percentage of sigma to add to the data, defaults to 10
        """
        self.__init_args_kwargs__ = copy.copy(
            locals()
        )  # save init args for recreation. not a deep copy to avoid unnecessary use of memory

        mapping = {
            "instance_random_seed": "-i",
            "number_of_classes": "-c",
            "number_of_attributes": "-a",
            "number_of_drifting_attributes": "-k",
            "magnitude_of_change": "-t",
            "noise_percentage": "-n",
            "sigma_percentage": "-s",
        }
        self.moa_stream = MOA_HyperplaneGenerator()
        config_str = build_cli_str_from_mapping_and_locals(mapping, locals())

        super().__init__(
            moa_stream=self.moa_stream,
            CLI=config_str,
        )

    # def __str__(self):
    # attributes = [
    #     (
    #         f"instance_random_seed={self.instance_random_seed}"
    #         if self.instance_random_seed != 1
    #         else None
    #     ),
    #     (
    #         f"number_of_classes={self.number_of_classes}"
    #         if self.number_of_classes != 2
    #         else None
    #     ),
    #     (
    #         f"number_of_attributes={self.number_of_attributes}"
    #         if self.number_of_attributes != 10
    #         else None
    #     ),
    #     (
    #         f"number_of_drifting_attributes={self.number_of_drifting_attributes}"
    #         if self.number_of_drifting_attributes != 2
    #         else None
    #     ),
    #     (
    #         f"magnitude_of_change={self.magnitude_of_change}"
    #         if self.magnitude_of_change != 0.0
    #         else None
    #     ),
    #     (
    #         f"noise_percentage={self.noise_percentage}"
    #         if self.noise_percentage != 5
    #         else None
    #     ),
    #     (
    #         f"sigma_percentage={self.sigma_percentage}"
    #         if self.sigma_percentage != 10
    #         else None
    #     ),
    # ]
    # non_default_attributes = [attr for attr in attributes if attr is not None]
    # return f"HyperPlaneClassification({', '.join(non_default_attributes)})"


class HyperPlaneRegression(MOAStream):
    """Generates HyperPlane Regression concepts functions.

    >>> from capymoa.stream.generator import HyperPlaneRegression
    ...
    >>> stream = HyperPlaneRegression()
    >>> stream.next_instance()
    RegressionInstance(
        Schema(generators.HyperplaneGeneratorForRegression ),
        x=[0.397 0.348 0.294 0.506 0.116 0.771 0.66  0.157 0.378 0.14 ],
        y_value=205.17965508540908
    )
    >>> stream.next_instance().x
    array([0.00485253, 0.85225356, 0.02341807, 0.70500995, 0.27502995,
           0.0753878 , 0.61059154, 0.95493077, 0.2740691 , 0.19020221])
    """

    def __init__(
        self,
        instance_random_seed: int = 1,
        number_of_classes: int = 2,
        number_of_attributes: int = 10,
        number_of_drifting_attributes: int = 2,
        magnitude_of_change: float = 0.0,
        noise_percentage: int = 5,
        sigma_percentage: int = 10,
    ):
        """Construct a HyperPlane Regression datastream generator.

        :param instance_random_seed: Seed for random generation of instances, defaults to 1
        :param number_of_classes: The number of classes of the generated instances, defaults to 2
        :param number_of_attributes: The number of attributes of the generated instances, defaults to 10
        :param number_of_drifting_attributes: The number of drifting attributes, defaults to 2
        :param magnitude_of_change: Magnitude of change in the generated instances, defaults to 0.0
        :param noise_percentage: Percentage of noise to add to the data, defaults to 10
        :param sigma_percentage: Percentage of sigma to add to the data, defaults to 10
        """
        self.__init_args_kwargs__ = copy.copy(
            locals()
        )  # save init args for recreation. not a deep copy to avoid unnecessary use of memory

        mapping = {
            "instance_random_seed": "-i",
            "number_of_classes": "-c",
            "number_of_attributes": "-a",
            "number_of_drifting_attributes": "-k",
            "magnitude_of_change": "-t",
            "noise_percentage": "-n",
            "sigma_percentage": "-s",
        }

        self.moa_stream = MOA_HyperplaneGeneratorForRegression()
        config_str = build_cli_str_from_mapping_and_locals(mapping, locals())

        super().__init__(CLI=config_str, moa_stream=self.moa_stream)

    # def __str__(self):
    #     attributes = [
    #         (
    #             f"instance_random_seed={self.instance_random_seed}"
    #             if self.instance_random_seed != 1
    #             else None
    #         ),
    #         (
    #             f"number_of_classes={self.number_of_classes}"
    #             if self.number_of_classes != 2
    #             else None
    #         ),
    #         (
    #             f"number_of_attributes={self.number_of_attributes}"
    #             if self.number_of_attributes != 10
    #             else None
    #         ),
    #         (
    #             f"number_of_drifting_attributes={self.number_of_drifting_attributes}"
    #             if self.number_of_drifting_attributes != 2
    #             else None
    #         ),
    #         (
    #             f"magnitude_of_change={self.magnitude_of_change}"
    #             if self.magnitude_of_change != 0.0
    #             else None
    #         ),
    #         (
    #             f"noise_percentage={self.noise_percentage}"
    #             if self.noise_percentage != 5
    #             else None
    #         ),
    #         (
    #             f"sigma_percentage={self.sigma_percentage}"
    #             if self.sigma_percentage != 10
    #             else None
    #         ),
    #     ]
    #     non_default_attributes = [attr for attr in attributes if attr is not None]
    #     return f"HyperPlaneRegression({', '.join(non_default_attributes)})"


class RandomRBFGenerator(MOAStream):
    """
    An Random RBF Generator

    >>> from capymoa.stream.generator import RandomRBFGenerator
    ...
    >>> stream = RandomRBFGenerator()
    >>> stream.next_instance()
    LabeledInstance(
        Schema(generators.RandomRBFGenerator ),
        x=[0.21  1.01  0.092 0.272 0.45  0.226 0.212 0.373 0.583 0.297],
        y_index=1,
        y_label='class2'
    )
    >>> stream.next_instance().x
    array([0.68807095, 0.62508298, 0.36161375, 0.29484898, 0.46067958,
           0.83491016, 0.69794979, 0.75702471, 0.79436834, 0.7605141 ])
    """

    def __init__(
        self,
        model_random_seed: int = 1,
        instance_random_seed: int = 1,
        number_of_classes: int = 2,
        number_of_attributes: int = 10,
        number_of_centroids: int = 50,
    ):
        """Construct a Random RBF Generator .

        :param instance_random_seed: Seed for random generation of instances, defaults to 1
        :param number_of_classes: The number of classes of the generated instances, defaults to 2
        :param number_of_attributes: The number of attributes of the generated instances, defaults to 10
        :param number_of_drifting_centroids: The number of drifting attributes, defaults to 2
        """

        mapping = {
            "model_random_seed": "-r",
            "instance_random_seed": "-i",
            "number_of_classes": "-c",
            "number_of_attributes": "-a",
            "number_of_centroids": "-n",
        }
        self.moa_stream = MOA_RandomRBFGenerator()
        config_str = build_cli_str_from_mapping_and_locals(mapping, locals())

        super().__init__(moa_stream=self.moa_stream, CLI=config_str)

    def __str__(self):
        attributes = [
            (
                f"model_random_seed={self.model_random_seed}"
                if self.model_random_seed != 1
                else None
            ),
            (
                f"instance_random_seed={self.instance_random_seed}"
                if self.instance_random_seed != 1
                else None
            ),
            (
                f"number_of_classes={self.number_of_classes}"
                if self.number_of_classes != 2
                else None
            ),
            (
                f"number_of_attributes={self.number_of_attributes}"
                if self.number_of_attributes != 10
                else None
            ),
            (
                f"number_of_centroids={self.number_of_centroids}"
                if self.number_of_centroids != 50
                else None
            ),
        ]
        non_default_attributes = [attr for attr in attributes if attr is not None]
        return f"RandomRBFGenerator({', '.join(non_default_attributes)})"


class RandomRBFGeneratorDrift(MOAStream):
    """
    Generates Random RBF concepts functions.
    """

    def __init__(
        self,
        model_random_seed: int = 1,
        instance_random_seed: int = 1,
        number_of_classes: int = 2,
        number_of_attributes: int = 10,
        number_of_centroids: int = 50,
        number_of_drifting_centroids: int = 2,
        magnitude_of_change: float = 0.0,
    ):
        """Construct a RBF Generator Classification/Clustering datastream generator.

        :param instance_random_seed: Seed for random generation of instances, defaults to 1
        :param number_of_classes: The number of classes of the generated instances, defaults to 2
        :param number_of_attributes: The number of attributes of the generated instances, defaults to 10
        :param number_of_drifting_centroids: The number of drifting attributes, defaults to 2
        :param magnitude_of_change: Magnitude of change in the generated instances, defaults to 0.0
        :param noise_percentage: Percentage of noise to add to the data, defaults to 10
        :param sigma_percentage: Percentage of sigma to add to the data, defaults to 10
        """

        mapping = {
            "model_random_seed": "-r",
            "instance_random_seed": "-i",
            "number_of_classes": "-c",
            "number_of_attributes": "-a",
            "number_of_centroids": "-n",
            "number_of_drifting_centroids": "-k",
            "magnitude_of_change": "-s",
        }
        self.moa_stream = MOA_RandomRBFGeneratorDrift()
        config_str = build_cli_str_from_mapping_and_locals(mapping, locals())

        super().__init__(moa_stream=self.moa_stream, CLI=config_str)

    def __str__(self):
        attributes = [
            (
                f"model_random_seed={self.model_random_seed}"
                if self.model_random_seed != 1
                else None
            ),
            (
                f"instance_random_seed={self.instance_random_seed}"
                if self.instance_random_seed != 1
                else None
            ),
            (
                f"number_of_classes={self.number_of_classes}"
                if self.number_of_classes != 2
                else None
            ),
            (
                f"number_of_attributes={self.number_of_attributes}"
                if self.number_of_attributes != 10
                else None
            ),
            (
                f"number_of_centroids={self.number_of_centroids}"
                if self.number_of_centroids != 50
                else None
            ),
            (
                f"number_of_drifting_centroids={self.number_of_drifting_centroids}"
                if self.number_of_drifting_centroids != 2
                else None
            ),
            (
                f"magnitude_of_change={self.magnitude_of_change}"
                if self.magnitude_of_change != 0.0
                else None
            ),
        ]
        non_default_attributes = [attr for attr in attributes if attr is not None]
        return f"RandomRBFGeneratorDrift({', '.join(non_default_attributes)})"


class AgrawalGenerator(MOAStream):
    """
    An Agrawal Generator

    >>> from capymoa.stream.generator import AgrawalGenerator
    ...
    >>> stream = AgrawalGenerator()
    >>> stream.next_instance()
    LabeledInstance(
        Schema(generators.AgrawalGenerator ),
        x=[1.105e+05 0.000e+00 5.400e+01 3.000e+00 1.400e+01 4.000e+00 1.350e+05
            3.000e+01 3.547e+05],
        y_index=1,
        y_label='groupB'
    )
    >>> stream.next_instance().x
    array([1.40893779e+05, 0.00000000e+00, 4.40000000e+01, 4.00000000e+00,
           1.90000000e+01, 7.00000000e+00, 1.35000000e+05, 2.00000000e+00,
           3.95015339e+05])
    """

    def __init__(
        self,
        instance_random_seed: int = 1,
        classification_function: int = 1,
        peturbation: float = 0.05,
        balance_classes: bool = False,
    ):
        """Construct an Agrawal Generator

        :param instance_random_seed: Seed for random generation of instances.
        :param classification_function: Classification function used, as defined in the original paper.
        :param peturbation: The amount of peturbation (noise) introduced to numeric values
        :param balance: Balance the number of instances of each class.
        """
        self.__init_args_kwargs__ = copy.copy(
            locals()
        )  # save init args for recreation. not a deep copy to avoid unnecessary use of memory

        self.moa_stream = MOA_AgrawalGenerator()

        self.instance_random_seed = instance_random_seed
        self.classification_function = classification_function
        self.peturbation = peturbation
        self.balance_classes = balance_classes

        self.CLI = f"-i {self.instance_random_seed} -f {self.classification_function} \
            -p {self.peturbation} {'-b' if self.balance_classes else ''}"

        super().__init__(CLI=self.CLI, moa_stream=self.moa_stream)

    def __str__(self):
        attributes = [
            (
                f"instance_random_seed={self.instance_random_seed}"
                if self.instance_random_seed != 1
                else None
            ),
            (f"classification_function={self.classification_function}"),
            (f"peturbation={self.peturbation}" if self.peturbation != 0.05 else None),
            (f"balance={self.balance}" if self.balance else None),
        ]

        non_default_attributes = [attr for attr in attributes if attr is not None]
        return f"AgrawalGenerator({', '.join(non_default_attributes)})"


class LEDGenerator(MOAStream):
    """
    An LED Generator

    >>> from capymoa.stream.generator import LEDGenerator
    ...
    >>> stream = LEDGenerator()
    >>> stream.next_instance()
    LabeledInstance(
        Schema(generators.LEDGenerator ),
        x=[1. 1. 0. ... 0. 0. 0.],
        y_index=5,
        y_label='5'
    )
    >>> stream.next_instance().x
    array([1., 1., 1., 0., 1., 1., 0., 0., 0., 1., 0., 1., 0., 1., 1., 0., 1.,
           0., 0., 1., 1., 0., 1., 1.])
    """

    def __init__(
        self,
        instance_random_seed: int = 1,
        noise_percentage: int = 10,
        reduce_data: bool = False,
    ):
        """Construct an LED Generator

        :param instance_random_seed: Seed for random generation of instances.
        :param noise_percentage: Percentage of noise to add to the data
        :param reduce_data: Reduce the data to only contain 7 relevant binary attributes
        """
        self.__init_args_kwargs__ = copy.copy(locals())
        # save init args for recreation. not a deep copy to avoid unnecessary use of memory

        self.moa_stream = MOA_LEDGenerator()

        self.instance_random_seed = instance_random_seed
        self.noise_percentage = noise_percentage
        self.reduce_data = reduce_data

        self.CLI = f"-i {self.instance_random_seed} -n {self.noise_percentage} \
            {'-s' if self.reduce_data else ''}"

        super().__init__(CLI=self.CLI, moa_stream=self.moa_stream)

    def __str__(self):
        attributes = [
            (
                f"instance_random_seed={self.instance_random_seed}"
                if self.instance_random_seed != 1
                else None
            ),
            (
                f"noise_percentage={self.noise_percentage}"
                if self.noise_percentage != 10
                else None
            ),
            (f"reduce_data={self.reduce_data}" if self.reduce_data else None),
        ]

        non_default_attributes = [attr for attr in attributes if attr is not None]
        return f"LEDGenerator({', '.join(non_default_attributes)})"


class LEDGeneratorDrift(MOAStream):
    """
    An LED Generator Drift

    >>> from capymoa.stream.generator import LEDGeneratorDrift
    ...
    >>> stream = LEDGeneratorDrift()
    >>> stream.next_instance()
    LabeledInstance(
        Schema(generators.LEDGeneratorDrift -d 7),
        x=[1. 1. 0. ... 0. 0. 0.],
        y_index=5,
        y_label='5'
    )
    >>> stream.next_instance().x
    array([0., 0., 1., 0., 1., 0., 1., 1., 1., 1., 0., 1., 1., 0., 1., 0., 1.,
           0., 0., 1., 1., 0., 1., 1.])
    """

    def __init__(
        self,
        instance_random_seed: int = 1,
        noise_percentage: int = 10,
        reduce_data: bool = False,
        number_of_attributes_with_drift: int = 7,
    ):
        """Construct an LED Generator Drift

        :param instance_random_seed: Seed for random generation of instances.
        :param noise_percentage: Percentage of noise to add to the data
        :param reduce_data: Reduce the data to only contain 7 relevant binary attributes
        :param number_of_attributes_with_drift: Number of attributes with drift
        """
        self.__init_args_kwargs__ = copy.copy(locals())
        # save init args for recreation. not a deep copy to avoid unnecessary use of memory

        self.moa_stream = MOA_LEDGeneratorDrift()

        self.instance_random_seed = instance_random_seed
        self.noise_percentage = noise_percentage
        self.reduce_data = reduce_data
        self.number_of_attributes_with_drift = number_of_attributes_with_drift

        self.CLI = f"-i {self.instance_random_seed} -n {self.noise_percentage} \
            {'-s' if self.reduce_data else ''} -d {self.number_of_attributes_with_drift}"

        super().__init__(CLI=self.CLI, moa_stream=self.moa_stream)

    def __str__(self):
        attributes = [
            (
                f"instance_random_seed={self.instance_random_seed}"
                if self.instance_random_seed != 1
                else None
            ),
            (
                f"noise_percentage={self.noise_percentage}"
                if self.percentage != 10
                else None
            ),
            (f"reduce_data={self.reduce_data}" if self.reduce_data else None)(
                f"number_of_attributes_with_drift={self.number_of_attributes_with_drift}"
                if self.number_of_attributes_with_drift != 7
                else None
            ),
        ]

        non_default_attributes = [attr for attr in attributes if attr is not None]
        return f"LEDGeneratorDrift({', '.join(non_default_attributes)})"


class WaveformGenerator(MOAStream):
    """
    An Waveform Generator

    >>> from capymoa.stream.generator import WaveformGenerator
    ...
    >>> stream = WaveformGenerator()
    >>> stream.next_instance()
    LabeledInstance(
        Schema(generators.WaveformGenerator ),
        x=[-0.092 -0.362  1.163 ...  2.825  0.765 -0.187],
        y_index=0,
        y_label='class1'
    )
    >>> stream.next_instance().x
    array([-0.35222814, -0.65631772,  1.66984311,  1.3552564 ,  1.95122954,
            3.34644007,  4.75457662,  2.72801084,  3.40907743,  2.41282297,
            3.34658027,  2.42282518,  2.08432716,  0.78783527,  0.94201874,
            0.75833533, -0.82178614, -1.23317608, -0.52710197, -0.44639196,
           -2.026593  ])
    """

    def __init__(
        self,
        instance_random_seed: int = 1,
        noise: bool = False,
    ):
        """Construct a WaveForm Generator .

        :param instance_random_seed: Seed for random generation of instances, defaults to 1
        :param noise: Adds noise for a total of 40 attributes
        """

        mapping = {
            "instance_random_seed": "-i",
            "noise": "-n",
        }
        self.moa_stream = MOA_WaveformGenerator()
        config_str = build_cli_str_from_mapping_and_locals(mapping, locals())

        super().__init__(moa_stream=self.moa_stream, CLI=config_str)

    def __str__(self):
        attributes = [
            (
                f"instance_random_seed={self.instance_random_seed}"
                if self.instance_random_seed != 1
                else None
            ),
            (f"noise={self.noise}" if self.noise else None),
        ]
        non_default_attributes = [attr for attr in attributes if attr is not None]
        return f"WaveformGenerator({', '.join(non_default_attributes)})"


class WaveformGeneratorDrift(MOAStream):
    """
    An Waveform Generator Drift

    >>> from capymoa.stream.generator import WaveformGeneratorDrift
    ...
    >>> stream = WaveformGeneratorDrift()
    >>> stream.next_instance()
    LabeledInstance(
        Schema(generators.WaveformGeneratorDrift -d 10),
        x=[3.787 3.658 5.136 ... 5.723 2.665 2.681],
        y_index=1,
        y_label='class2'
    )
    >>> stream.next_instance().x
    array([ 0.54985074,  2.17089406,  0.6142235 ,  3.18809944, -1.81293483,
           -0.11717947, -1.77198821, -0.14927903, -0.49779111, -1.33272998,
           -0.38139892, -1.49682927,  1.49204371,  2.65344343,  4.25116434,
            3.39751393,  2.90259886,  4.21403878,  1.98411715,  3.33956917,
            4.08153654])
    """

    def __init__(
        self,
        instance_random_seed: int = 1,
        noise: bool = False,
        number_of_attributes_with_drift: int = 10,
    ):
        """Construct a WaveformGeneratorDrift Generator .

        :param instance_random_seed: Seed for random generation of instances, defaults to 1
        :param noise: Adds noise for a total of 40 attributes
        :param number_of_attributes_with_drift: Number of attributes with drift
        """

        mapping = {
            "instance_random_seed": "-i",
            "noise": "-n",
            "number_of_attributes_with_drift": "-d",
        }
        self.moa_stream = MOA_WaveformGeneratorDrift()
        config_str = build_cli_str_from_mapping_and_locals(mapping, locals())

        super().__init__(moa_stream=self.moa_stream, CLI=config_str)

    def __str__(self):
        attributes = [
            (
                f"instance_random_seed={self.instance_random_seed}"
                if self.instance_random_seed != 1
                else None
            ),
            (f"noise={self.noise}" if self.noise else None),
            (
                f"number_of_attributes_with_drift={self.number_of_attributes_with_drift}"
                if self.number_of_attributes_with_drift != 10
                else None
            ),
        ]
        non_default_attributes = [attr for attr in attributes if attr is not None]
        return f"WaveformGeneratorDrift({', '.join(non_default_attributes)})"


class STAGGERGenerator(MOAStream):
    """
    An STAGGER Generator

    >>> from capymoa.stream.generator import STAGGERGenerator
    ...
    >>> stream = STAGGERGenerator()
    >>> stream.next_instance()
    LabeledInstance(
        Schema(generators.STAGGERGenerator ),
        x=[0. 1. 1.],
        y_index=0,
        y_label='false'
    )
    >>> stream.next_instance().x
    array([0., 2., 1.])
    """

    def __init__(
        self,
        instance_random_seed: int = 1,
        classification_function: int = 1,
        balance_classes: bool = False,
    ):
        """Construct a STAGGER Generator .

        :param instance_random_seed: Seed for random generation of instances, defaults to 1
        :param classification_function: Classification function used, as defined in the original paper.
        :param balance: Balance the number of instances of each class.
        """

        mapping = {
            "instance_random_seed": "-i",
            "classification_function": "-f",
            "balance_classes": "-b",
        }
        self.moa_stream = MOA_STAGGERGenerator()
        config_str = build_cli_str_from_mapping_and_locals(mapping, locals())

        super().__init__(moa_stream=self.moa_stream, CLI=config_str)

    def __str__(self):
        attributes = [
            (
                f"instance_random_seed={self.instance_random_seed}"
                if self.instance_random_seed != 1
                else None
            ),
            (
                f"classification_function={self.classification_function}"
                if self.classification_function != 1
                else None
            ),
            (
                f"balance_classes={self.balance_classes}"
                if self.balance_classes
                else None
            ),
        ]
        non_default_attributes = [attr for attr in attributes if attr is not None]
        return f"STAGGERGenerator({', '.join(non_default_attributes)})"


class SineGenerator(MOAStream):
    """
    An SineGenerator

    >>> from capymoa.stream.generator import SineGenerator
    ...
    >>> stream = SineGenerator()
    >>> stream.next_instance()
    LabeledInstance(
        Schema(generators.SineGenerator ),
        x=[0.731 0.41  0.208 0.333],
        y_index=0,
        y_label='positive'
    )
    >>> stream.next_instance().x
    array([0.96775591, 0.00611718, 0.9637048 , 0.93986539])
    """

    def __init__(
        self,
        instance_random_seed: int = 1,
        classification_function: int = 1,
        suppress_irrelevant_attributes: bool = False,
        balance_classes: bool = False,
    ):
        """Construct a SineGenerator .

        :param instance_random_seed: Seed for random generation of instances, defaults to 1
        :param classification_function: Classification function used, as defined in the original paper.
        :param suppress_irrelevant_attributes: Reduce the data to only contain 2 relevant numeric attributes
        :param balance: Balance the number of instances of each class.
        """

        mapping = {
            "instance_random_seed": "-i",
            "classification_function": "-f",
            "suppress_irrelevant_attributes": "-s",
            "balance_classes": "-b",
        }
        self.moa_stream = MOA_SineGenerator()
        config_str = build_cli_str_from_mapping_and_locals(mapping, locals())

        super().__init__(moa_stream=self.moa_stream, CLI=config_str)

    def __str__(self):
        attributes = [
            (
                f"instance_random_seed={self.instance_random_seed}"
                if self.instance_random_seed != 1
                else None
            ),
            (
                f"classification_function={self.classification_function}"
                if self.classification_function != 1
                else None
            ),
            (
                f"suppress_irrelevant_attributes={self.suppress_irrelevant_attributes}"
                if self.suppress_irrelevant_attributes
                else None
            ),
            (
                f"balance_classes={self.balance_classes}"
                if self.balance_classes
                else None
            ),
        ]
        non_default_attributes = [attr for attr in attributes if attr is not None]
        return f"SineGenerator({', '.join(non_default_attributes)})"
