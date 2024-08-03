"""Generate artificial data streams."""
import copy

from capymoa.stream import Stream
from capymoa.stream._stream import Schema
from moa.streams import InstanceStream
from moa.streams.generators import RandomTreeGenerator as MOA_RandomTreeGenerator
from moa.streams.generators import SEAGenerator as MOA_SEAGenerator
from moa.streams.generators import HyperplaneGenerator as MOA_HyperplaneGenerator
from moa.streams.generators import HyperplaneGeneratorForRegression as MOA_HyperplaneGeneratorForRegression
from moa.streams.generators import RandomRBFGeneratorDrift as MOA_RandomRBFGeneratorDrift
from moa.streams.generators import AgrawalGenerator as MOA_AgrawalGenerator
from moa.streams.generators import LEDGenerator as MOA_LEDGenerator
from capymoa._utils import build_cli_str_from_mapping_and_locals



class RandomTreeGenerator(Stream):
    """Stream generator for a stream based on a randomly generated tree.

    >>> from capymoa.stream.generator import RandomTreeGenerator
    ...
    >>> stream = RandomTreeGenerator()
    >>> stream.next_instance()
    LabeledInstance(
        Schema(generators.RandomTreeGenerator ),
        x=ndarray(..., 10),
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
        self.__init_args_kwargs__ = copy.copy(locals())  # save init args for recreation. not a deep copy to avoid unnecessary use of memory

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


class SEA(Stream):
    """Generates SEA concepts functions.

    >>> from capymoa.stream.generator import SEA
    ...
    >>> stream = SEA()
    >>> stream.next_instance()
    LabeledInstance(
        Schema(generators.SEAGenerator ),
        x=ndarray(..., 3),
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
        self.__init_args_kwargs__ = copy.copy(locals())  # save init args for recreation. not a deep copy to avoid unnecessary use of memory

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


class HyperPlaneClassification(Stream):
    """Generates HyperPlane concepts functions.

    >>> from capymoa.stream.generator import HyperPlaneClassification
    ...
    >>> stream = HyperPlaneClassification()
    >>> stream.next_instance()
    LabeledInstance(
        Schema(generators.HyperplaneGenerator ),
        x=ndarray(..., 10),
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
        self.__init_args_kwargs__ = copy.copy(locals())  # save init args for recreation. not a deep copy to avoid unnecessary use of memory

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


class HyperPlaneRegression(Stream):
    """Generates HyperPlane Regression concepts functions.

    >>> from capymoa.stream.generator import HyperPlaneRegression
    ...
    >>> stream = HyperPlaneRegression()
    >>> stream.next_instance()
    RegressionInstance(
        Schema(generators.HyperplaneGeneratorForRegression ),
        x=ndarray(..., 10),
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
        self.__init_args_kwargs__ = copy.copy(locals())  # save init args for recreation. not a deep copy to avoid unnecessary use of memory

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

class RandomRBFGeneratorDrift(Stream):
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
            magnitude_of_change: float = 0.0
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
            "magnitude_of_change": "-s"            
        }
        self.moa_stream = MOA_RandomRBFGeneratorDrift()
        config_str = build_cli_str_from_mapping_and_locals(mapping, locals())


        super().__init__(
            moa_stream=self.moa_stream,
            CLI=config_str
        )

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
    

class AgrawalGenerator(Stream):
    """
    An Agrawal Generator

    >>> from capymoa.stream.generator import AgrawalGenerator
    ...
    >>> stream = AgrawalGenerator()
    >>> stream.next_instance()
    LabeledInstance(
        Schema(generators.AgrawalGenerator ),
        x=ndarray(..., 10),
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
            classification_function: int = 1,
            peturbation: int = 1,
            balance: int = 1
    ):
        """ Construct an Agrawal Generator

        :param instance_random_seed: Seed for random generation of instances.
        :param classification_function: Classification function used, as defined in the original paper.
        :param peturbation: The amount of peturbation (noise) introduced to numeric values
        :param balance: Balance the number of instances of each class.
        """
        self.__init_args_kwargs__ = copy.copy(locals())  # save init args for recreation. not a deep copy to avoid unnecessary use of memory
        
        self.moa_stream = MOA_AgrawalGenerator()

        self.instance_random_seed = instance_random_seed
        self.classification_function = classification_function
        self.peturbation = peturbation
        self.balance = balance

        self.CLI = f"-i {self.instance_random_seed} -f {self.classification_function} \
            -p {self.peturbation} -b {self.balance}"
    
        super().__init__(CLI=self.CLI, moa_stream=self.moa_stream)


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
                f"peturbation={self.peturbation}"
                if self.peturbation != 1
                else None
            ),
            (
                f"balance={self.balance}"
                if self.balance != 1
                else None
            )
        ]

        non_default_attributes = [attr for attr in attributes if attr is not None]
        return f"AgrawalGenerator({', '.join(non_default_attributes)})"
    

class LEDGenerator(Stream):
    """
    An LED Generator

    >>> from capymoa.stream.generator import LEDGenerator
    ...
    >>> stream = LEDGenerator()
    >>> stream.next_instance()
    LabeledInstance(
        Schema(generators.LEDGenerator ),
        x=ndarray(..., 10),
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
            percentage: int = 1,
            reduce_data: bool = False,
    ):
        """ Construct an LED Generator

        :param instance_random_seed: Seed for random generation of instances.
        :param percentage: Percentage of noise to add to the data
        :param reduce_data: Reduce the data to only contain 7 relevant binary attributes
         """
        self.__init_args_kwargs__ = copy.copy(locals())  # save init args for recreation. not a deep copy to avoid unnecessary use of memory
        
        self.moa_stream = MOA_LEDGenerator()

        self.instance_random_seed = instance_random_seed
        self.percentage = percentage
        self.reduce_data = reduce_data
        
        self.CLI = f"-i {self.instance_random_seed} -n {self.percentage} \
            -s {self.reduce_data}"
    
        super().__init__(CLI=self.CLI, moa_stream=self.moa_stream)


    def __str__(self):
        attributes = [
            (
                f"instance_random_seed={self.instance_random_seed}"
                if self.instance_random_seed != 1
                else None
            ),
            (
                f"percentage={self.percentage}"
                if self.percentage != 1
                else None
            ),
            (
                f"reduce_data={self.reduce_data}"
                if self.reduce_data != False
                else None
            )
        ]

        non_default_attributes = [attr for attr in attributes if attr is not None]
        return f"LEDGenerator({', '.join(non_default_attributes)})"