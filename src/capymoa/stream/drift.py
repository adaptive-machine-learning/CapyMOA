"""Simulate concept drift in datastreams."""
import copy
import inspect
import re
from collections import OrderedDict
from itertools import cycle

from capymoa.stream._stream import Stream
from capymoa._utils import _get_moa_creation_CLI
from moa.streams import ConceptDriftStream as MOA_ConceptDriftStream


class DriftStream(Stream):
    def __init__(self, schema=None, CLI=None, moa_stream=None, stream=None):
        # If moa_stream is specified, just instantiate it directly. We can check whether it is a ConceptDriftStream object or not.
        # if composite_stream is set, then the ConceptDriftStream object is build according to the list of Concepts and Drifts specified in composite_stream
        # ```moa_stream``` and ```CLI``` allow the user to specify the stream using a ConceptDriftStream from MOA alongside its CLI. However, in the future we might remove that functionality to make the code simpler.

        self.stream = stream
        self.drifts = []
        moa_concept_drift_stream = MOA_ConceptDriftStream()

        if CLI is None:
            stream1 = None
            stream2 = None
            drift = None

            CLI = ""
            for component in self.stream:
                if isinstance(component, Stream):
                    if stream1 is None:
                        stream1 = component
                    else:
                        stream2 = component
                        if drift is None:
                            raise ValueError(
                                "A Drift object must be specified between two Stream objects."
                            )

                        CLI += f" -d {_get_moa_creation_CLI(stream2.moa_stream)} -w {drift.width} -p {drift.position} -r {drift.random_seed} -a {drift.alpha}"
                        CLI = CLI.replace(
                            "streams.", ""
                        )  # got to remove package name from streams.ConceptDriftStream

                        stream1 = Stream(moa_stream=moa_concept_drift_stream, CLI=CLI)
                        stream2 = None

                elif isinstance(component, Drift):
                    # print(component)
                    drift = component
                    self.drifts.append(drift)
                    CLI = f" -s {_get_moa_creation_CLI(stream1.moa_stream)} "

            # print(CLI)
            # CLI = command_line
            moa_stream = moa_concept_drift_stream
        else:
            # [EXPERIMENTAL]
            # If the user is attempting to create a DriftStream using a MOA CLI, we need to derive the Drift meta-data through the CLI.
            # The number of ConceptDriftStream occurrences corresponds to the number of Drifts.
            # +1 because we expect at least one drift from an implit ConceptDriftStream (i.e. not shown in the CLI because it is the moa_stream object)
            num_drifts = CLI.count("ConceptDriftStream") + 1

            # This is a best effort in obtaining the meta-data from a MOA CLI.
            # Notice that if the width (-w) or position (-p) are not explicitly shown in the CLI it is difficult to infer them.
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
                    # Assuming the width of the drifts (or at least one) are not show, implies that the default value (1000) was used.
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
        # If the stream was defined using the backward compatility (MOA object + CLI) then there are no Stream objects in stream.
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
        self.__init_args_kwargs__ = copy.copy(locals())  # save init args for recreation. not a deep copy to avoid unnecessary use of memory
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
        self.__init_args_kwargs__ = copy.copy(locals())  # save init args for recreation. not a deep copy to avoid unnecessary use of memory

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
    init_args.pop('self') # remove self item

    # get instance's values for __init__
    # assumes attribute name is same as parameter name
    # args = {attr: getattr(obj, attr) for attr in init_args}
    args = {attr: obj.__init_args_kwargs__[attr] for attr in init_args}
    return cls, args


def get_recurrent_concept_drift_stream_list(
        concept_list: list,
        max_recurrences_per_concept: int = 3,
        change_random_seed_for_same_concept: bool = False,
        transition_type_template: Drift = AbruptDrift(position=5000),
        concept_name_list: list = None,
) -> list:
    recurrent_drift_stream = []
    recurrent_concept_info = []

    if not isinstance(transition_type_template, AbruptDrift):
        raise ValueError(f'Unsupported drift transition type: {str(transition_type_template)}')

    concept_cycle = IndexedCycle([k for k in concept_list])

    recurrence_count = {i: 0 for i, v in enumerate(concept_list)}

    drift_cls, original_drift_args = get_class_and_init_attributes_with_values(transition_type_template)

    max_concepts = len(concept_list) * max_recurrences_per_concept
    start_of_concept = 0
    for i in range(0, max_concepts * 2, 2): # get even indexes starting from 0
        next_idx, next_concept = next(concept_cycle) # get next concept and its index
        drift_args = copy.deepcopy(original_drift_args) # create a copy of drift args
        position = (int)(drift_args['position'] * ((i + 2) / 2)) # calculate drift position
        drift_args['position'] = position # set drift position
        drift = drift_cls(**drift_args) # initialize drift

        end_of_concept = position
        add_concept = False

        if recurrence_count[next_idx] < max_recurrences_per_concept:
            stream_cls, strea_args = get_class_and_init_attributes_with_values(next_concept)
            add_concept = True
        # else:  # recurrence recurrence_count has exceeded

        if add_concept:
            recurrent_drift_stream.insert(i, stream_cls(**strea_args))
            recurrence_count[next_idx] += 1
            recurrent_drift_stream.insert(i + 1, drift)
            stream_name = f'concept {next_idx}' if concept_name_list is None else concept_name_list[next_idx]
            recurrent_concept_info.append({'id': stream_name, 'start': start_of_concept, 'end': end_of_concept})
            start_of_concept = end_of_concept
            end_of_concept = None

    recurrent_drift_stream.pop(len(recurrent_drift_stream) - 1) # remove last Drift item

    return recurrent_concept_info, recurrent_drift_stream


class RecurrentConceptDriftStream(DriftStream):
    def __init__(self,
                 concept_list: list,
                 max_recurrences_per_concept: int = 3,
                 change_random_seed_for_same_concept: bool = False,
                 transition_type_template: Drift = AbruptDrift(position=5000),
                 concept_name_list: list = None
                 ):
        self.recurrent_concept_info, stream_list = get_recurrent_concept_drift_stream_list(
            concept_list=concept_list,
            max_recurrences_per_concept=max_recurrences_per_concept,
            change_random_seed_for_same_concept=change_random_seed_for_same_concept,
            transition_type_template=transition_type_template,
            concept_name_list = concept_name_list
        )

        super().__init__(stream=stream_list)
