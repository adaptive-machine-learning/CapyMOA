"""Simulate concept drift in datastreams."""

import copy
import inspect
import re
from collections import OrderedDict
from itertools import cycle

from capymoa.stream import MOAStream
from capymoa._utils import _get_moa_creation_CLI
from moa.streams import ConceptDriftStream as MOA_ConceptDriftStream


class DriftStream(MOAStream):
    def __init__(self, schema=None, CLI=None, moa_stream=None, stream=None):
        """
        Initialize the stream with the specified parameters.

        :param schema: The schema that defines the structure of the stream. Default is None.
        :param CLI: Command Line Interface string used to define the stream configuration
            for the MOA (Massive Online Analysis) framework. Default is None.
        :param moa_stream: A pre-configured ConceptDriftStream object from MOA. If specified,
            the stream will be instantiated directly using this object. This is useful
            for integrating with MOA-based streams. Default is None.
        :param stream: A list that defines a composite stream consisting of various concepts
            and drifts. If this is set, the ConceptDriftStream object will be built according
            to the list of concepts and drifts specified. Default is None.

        Notes:
        ------
        - If `moa_stream` is specified, it takes precedence, and the stream will be
          instantiated directly from the provided ConceptDriftStream object.

        - The `CLI` and `moa_stream` parameters allow users to specify the stream
          using a ConceptDriftStream from MOA alongside its CLI. This provides
          flexibility for users familiar with MOA's configuration style.

        - In the future, we might remove the functionality associated with `CLI`
          and `moa_stream` to simplify the code, focusing on other methods of stream
          configuration.
        """
        self.stream = stream
        self.drifts = []

        if CLI is None:
            stream1 = None
            stream2 = None
            drift = None

            CLI = ""
            for component in self.stream:
                if isinstance(component, MOAStream):
                    if stream1 is None:
                        stream1 = component
                    else:
                        stream2 = component
                        if drift is None:
                            raise ValueError(
                                "A Drift object must be specified between two Stream objects."
                            )

                        CLI += (
                            f" -d {_get_moa_creation_CLI(stream2.moa_stream)} -w {drift.width} -p "
                            f"{drift.position} -r {drift.random_seed} -a {drift.alpha}"
                        )
                        CLI = CLI.replace(
                            "streams.", ""
                        )  # got to remove package name from streams.ConceptDriftStream

                        stream1 = MOAStream(
                            moa_stream=MOA_ConceptDriftStream(), CLI=CLI
                        )
                        stream2 = None

                elif isinstance(component, Drift):
                    # print(component)
                    drift = component
                    self.drifts.append(drift)
                    CLI = f" -s {_get_moa_creation_CLI(stream1.moa_stream)} "

            moa_stream = MOA_ConceptDriftStream()
        else:
            # [EXPERIMENTAL]
            # If the user is attempting to create a DriftStream using a MOA CLI, we need to derive the Drift meta-data
            # through the CLI. The number of ConceptDriftStream occurrences corresponds to the number of Drifts.
            # +1 because we expect at least one drift from an implicit ConceptDriftStream (i.e. not shown in the CLI
            # because it is the moa_stream object)
            num_drifts = CLI.count("ConceptDriftStream") + 1

            # This is a best effort in obtaining the meta-data from a MOA CLI.
            # Notice that if the width (-w) or position (-p) are not explicitly shown in the CLI it is difficult to
            # infer them.
            pattern_position = r"-p (\d+)"
            pattern_width = r"-w (\d+)"
            matches_position = re.findall(pattern_position, CLI)
            matches_width = re.findall(pattern_width, CLI)

            for i in range(0, num_drifts):
                if len(matches_width) == len(matches_position):
                    self.drifts.append(
                        Drift(
                            position=int(matches_position[i]),
                            width=int(matches_width[i]),
                        )
                    )
                else:
                    # Assuming the width of the drifts (or at least one) are not show, implies that the default
                    # value (1000) was used.
                    self.drifts.append(
                        Drift(position=int(matches_position[i]), width=1000)
                    )

        super().__init__(schema=schema, CLI=CLI, moa_stream=moa_stream)

    def get_num_drifts(self):
        return len(self.drifts)

    def get_drifts(self):
        return self.drifts

    def __str__(self):
        if self.stream is not None:
            return ",".join(str(component) for component in self.stream)
        # If the stream was defined using the backward compatility (MOA object + CLI) then there are no Stream
        # objects in stream.
        # Best we can do is return the CLI directly.
        return f"ConceptDriftStream {self._CLI}"


# TODO: remove width from the base Drift class and keep it only on the GradualDrift


class Drift:
    """Represents a concept drift in a DriftStream. See 2.7.1 Concept drift framework in [1]_.

    .. [1] Bifet, Albert, et al. "Data stream mining: a practical approach." COSI (2011).
    """

    def __init__(self, position, width=0, alpha=0.0, random_seed=1):
        """Construct a drift in a DriftStream.

        :param position: The location of the drift in terms of the number of instances processed prior to it occurring.
        :param width: The size of the window of change. A width of 0 or 1 corresponds to an abrupt drift.
        :param alpha: The grade of change, defaults to 0.0.
        :param random_seed: Seed for random number generation, defaults to 1.
        """
        self.width = width
        self.position = position
        self.alpha = alpha
        self.random_seed = random_seed

    def __str__(self):
        drift_kind = "GradualDrift"
        if self.width == 0 or self.width == 1:
            drift_kind = "AbruptDrift"
        attributes = [
            f"position={self.position}",
            f"width={self.width}" if self.width not in [0, 1] else None,
            f"alpha={self.alpha}" if self.alpha != 0.0 else None,
            f"random_seed={self.random_seed}" if self.random_seed != 1 else None,
        ]
        non_default_attributes = [attr for attr in attributes if attr is not None]
        return f"{drift_kind}({', '.join(non_default_attributes)})"


class GradualDrift(Drift):
    def __init__(
        self, position=None, width=None, start=None, end=None, alpha=0.0, random_seed=1
    ):
        self.__init_args_kwargs__ = copy.copy(
            locals()
        )  # save init args for recreation. not a deep copy to avoid unnecessary use of memory
        # since python doesn't allow overloading functions we need to check if the user hasn't defined position + width and start+end.
        if (
            position is not None
            and width is not None
            and start is not None
            and end is not None
        ):
            raise ValueError(
                "Either use start and end OR position and width to determine the location of the gradual drift."
            )

        if start is None and end is None:
            self.width = width
            self.position = position
            self.start = int(position - width / 2)
            self.end = int(position + width / 2)
        elif position is None and width is None:
            self.start = start
            self.end = end
            self.width = end - start
            print(width)
            self.position = int((start + end) / 2)

        self.alpha = alpha
        self.random_seed = random_seed

        super().__init__(
            position=self.position, random_seed=self.random_seed, width=self.width
        )

    def __str__(self):
        attributes = [
            f"position={self.position}",
            f"start={self.start}",
            f"end={self.end}",
            f"width={self.width}",
            f"alpha={self.alpha}" if self.alpha != 0.0 else None,
            f"random_seed={self.random_seed}" if self.random_seed != 1 else None,
        ]
        non_default_attributes = [attr for attr in attributes if attr is not None]
        return f"GradualDrift({', '.join(non_default_attributes)})"


class AbruptDrift(Drift):
    def __init__(self, position: int, random_seed: int = 1):
        self.__init_args_kwargs__ = copy.copy(
            locals()
        )  # save init args for recreation. not a deep copy to avoid unnecessary use of memory

        self.position = position
        self.random_seed = random_seed

        super().__init__(position=position, random_seed=random_seed)

    def __str__(self):
        attributes = [
            f"position={self.position}",
            f"random_seed={self.random_seed}" if self.random_seed != 1 else None,
        ]
        non_default_attributes = [attr for attr in attributes if attr is not None]
        return f"AbruptDrift({', '.join(non_default_attributes)})"


class IndexedCycle:
    """
    An iterator that cycles through an iterable, returning tuples of (index, item).

    Provides methods for replacing items at specific indices and resetting the cycle.
    """

    def __init__(self, iterable):
        self._data = list(iterable)  # Create a copy for modification
        self._index = -1
        self._cycle = cycle(self._data)

    def __iter__(self):
        return self

    def __next__(self):
        if self._index + 1 < len(self._data):
            self._index += 1
        else:
            self._index = 0
        item = next(self._cycle)

        return self._index, item

    def replace_and_move_to(self, index, new_item):
        """
        Replaces the item at the specified index in the original data and moves it to the new position.

        Raises:
            ValueError: If the index is out of range.
        """

        if not 0 <= index < len(self._data):
            raise ValueError("Index out of range")

        self._data[index] = new_item
        self._cycle = cycle(self._data)  # Reset the cycle with modified data

        # move to new index
        while self._index != index:
            _, _ = self.__next__()


def get_class_and_init_attributes_with_values(obj):
    cls = type(obj)
    # Get function signature of init
    function_signature = inspect.signature(cls.__init__)
    # Access parameter names
    init_args = OrderedDict(function_signature.parameters)
    init_args.pop("self")  # remove self item

    # get instance's values for __init__
    # assumes attribute name is same as parameter name
    # args = {attr: getattr(obj, attr) for attr in init_args}
    args = {attr: obj.__init_args_kwargs__[attr] for attr in init_args}
    return cls, args


def get_recurrent_concept_drift_stream_list(
    concept_list: list,
    max_recurrences_per_concept: int = 3,
    transition_type_template: Drift = AbruptDrift(position=5000),
    concept_name_list: list = None,
) -> list:
    # checks
    if not isinstance(transition_type_template, AbruptDrift):
        raise ValueError(
            f"Unsupported drift transition type: {str(transition_type_template)}"
        )

    # variable initializations
    concept_cycle = IndexedCycle([k for k in concept_list])
    drift_stream = []
    concept_info = []
    recurrent_concept_info = {
        i: {"count": 0, "instance_random_seed": None}
        for i, v in enumerate(concept_list)
    }

    # get drift args and class
    drift_cls, original_drift_args = get_class_and_init_attributes_with_values(
        transition_type_template
    )

    max_concepts = len(concept_list) * max_recurrences_per_concept
    start_of_concept = 0

    for i in range(0, max_concepts * 2, 2):  # get even indexes starting from 0
        # get next concept and its index
        next_concept_idx, next_concept = next(concept_cycle)

        # calculate and set drift position
        drift_args = copy.deepcopy(original_drift_args)  # create a copy of drift args
        position = (int)(
            drift_args["position"] * ((i + 2) / 2)
        )  # calculate drift position
        drift_args["position"] = position  # set drift position
        drift = drift_cls(**drift_args)  # initialize drift

        end_of_concept = position
        add_concept = False

        if (
            recurrent_concept_info[next_concept_idx]["count"]
            < max_recurrences_per_concept
        ):
            stream_cls, original_stream_args = (
                get_class_and_init_attributes_with_values(next_concept)
            )
            stream_args = copy.deepcopy(
                original_stream_args
            )  # create a copy of stream args
            if (
                recurrent_concept_info[next_concept_idx]["instance_random_seed"] is None
            ):  # first iteration of this concept
                if not isinstance(
                    stream_args["instance_random_seed"], int
                ):  # probably 'instance_random_seed' not set
                    stream_args["instance_random_seed"] = 1
                recurrent_concept_info[next_concept_idx]["instance_random_seed"] = (
                    stream_args["instance_random_seed"]
                )
            else:  # not the first iteration of this concept
                recurrent_concept_info[next_concept_idx]["instance_random_seed"] += 1
            add_concept = True
        # else:  # recurrence concept_info has exceeded

        if add_concept:
            # update internal count and instance_random_seed
            stream_args["instance_random_seed"] = recurrent_concept_info[
                next_concept_idx
            ]["instance_random_seed"]
            recurrent_concept_info[next_concept_idx]["count"] += 1

            # add stream and drift to the list
            drift_stream.insert(i, stream_cls(**stream_args))
            drift_stream.insert(i + 1, drift)

            # generate concept info for plotting
            stream_name = (
                f"concept {next_concept_idx}"
                if concept_name_list is None
                else concept_name_list[next_concept_idx]
            )
            concept_info.append(
                {"id": stream_name, "start": start_of_concept, "end": end_of_concept}
            )
            start_of_concept = end_of_concept
            end_of_concept = None

    drift_stream.pop(len(drift_stream) - 1)  # remove last Drift item

    return concept_info, drift_stream


class RecurrentConceptDriftStream(DriftStream):
    def __init__(
        self,
        concept_list: list,
        max_recurrences_per_concept: int = 3,
        transition_type_template: Drift = AbruptDrift(position=5000),
        concept_name_list: list = None,
    ):
        self.concept_info, stream_list = get_recurrent_concept_drift_stream_list(
            concept_list=concept_list,
            max_recurrences_per_concept=max_recurrences_per_concept,
            transition_type_template=transition_type_template,
            concept_name_list=concept_name_list,
        )

        super().__init__(stream=stream_list)
