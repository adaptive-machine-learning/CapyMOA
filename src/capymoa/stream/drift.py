"""Simulate concept drift in datastreams."""

import copy
import inspect
import math
import random as _random
import re
from collections import OrderedDict
from itertools import cycle

from capymoa.stream import MOAStream, Stream
from capymoa._cli import cli_str_stream
from moa.streams import ConceptDriftStream as MOA_ConceptDriftStream


class _ConceptNode:
    """Leaf of the composition tree: draws from a single concept."""

    def __init__(self, stream):
        self.stream = stream

    def next_instance(self):
        return self.stream.next_instance()

    def has_more_instances(self):
        return self.stream.has_more_instances()

    def restart(self):
        self.stream.restart()


class _Transition:
    """Mixes two nodes across a drift.

    Mirrors the structure MOA builds by nesting ``ConceptDriftStream``: each
    node keeps its own instance counter and its own random generator, and only
    advances when it is actually drawn from.
    """

    def __init__(self, before, after, drift):
        self.before = before
        self.after = after
        self.drift = drift
        self.restart()

    def restart(self):
        self._n = 0
        self._rng = _random.Random(self.drift.random_seed)
        self.before.restart()
        self.after.restart()

    def probability_of_new_concept(self, n):
        """Probability an instance at ``n`` is drawn from the new concept.

        The default is the logistic ramp MOA uses,
        ``1 / (1 + exp(-4 (n - position) / width))``. A width of zero is a step
        at ``position`` -- MOA reaches the same result by dividing by zero and
        letting the exponential saturate.
        """
        width = self.drift.width
        if not width:
            return 1.0 if n >= self.drift.position else 0.0
        x = -4.0 * (n - self.drift.position) / width
        if x > 700:  # exp overflows; the ramp has saturated either way
            return 0.0
        if x < -700:
            return 1.0
        return 1.0 / (1.0 + math.exp(x))

    def next_instance(self):
        self._n += 1
        if self._rng.random() > self.probability_of_new_concept(self._n):
            return self.before.next_instance()
        return self.after.next_instance()

    def has_more_instances(self):
        # Only the concepts that could still be drawn from matter. Requiring
        # both would stop the stream early once a finite concept the drift has
        # already moved past runs out.
        probability = self.probability_of_new_concept(self._n + 1)
        if probability <= 0.0:
            return self.before.has_more_instances()
        if probability >= 1.0:
            return self.after.has_more_instances()
        # Mid-transition either concept can be selected for the next instance.
        return self.before.has_more_instances() and self.after.has_more_instances()


class DriftStream(Stream):
    """A stream composed of concepts separated by drifts.

    The stream is defined as a list that alternates concepts and drifts,
    starting and ending with a concept:

    >>> from capymoa.stream.drift import DriftStream, AbruptDrift, GradualDrift
    >>> from capymoa.stream.generator import SEA
    >>> stream = DriftStream(stream=[
    ...     SEA(function=1),
    ...     AbruptDrift(position=5000),
    ...     SEA(function=3),
    ...     GradualDrift(position=10000, width=2000),
    ...     SEA(function=1),
    ... ])
    >>> stream.get_num_drifts()
    2

    Composition happens in Python: instances are drawn from the concept
    selected for the current position, and across a :class:`GradualDrift` the
    choice is a per-instance draw against the transition ramp. Earlier versions
    delegated this to MOA's ``ConceptDriftStream``, which meant every concept
    had to be a MOA-backed stream.

    ``DriftStream`` is therefore a plain :class:`~capymoa.stream.Stream` rather
    than a :class:`~capymoa.stream.MOAStream`. Where a MOA object is genuinely
    needed -- to hand the stream to MOA code, or to print a MOA CLI -- use
    :func:`to_moa_stream`, which builds the equivalent nested
    ``ConceptDriftStream``. That conversion needs every concept to be
    MOA-backed, so it is offered explicitly instead of being the implementation.
    """

    def __init__(self, schema=None, CLI=None, moa_stream=None, stream=None):
        """Initialize the stream.

        :param schema: The schema of the stream. Taken from the first concept
            when not given.
        :param CLI: Command Line Interface string describing a MOA
            ``ConceptDriftStream``. Kept for backward compatibility; the stream
            is then backed by MOA rather than composed in Python.
        :param moa_stream: A pre-configured ``ConceptDriftStream`` from MOA,
            used together with ``CLI``.
        :param stream: The list of concepts and drifts to compose, alternating
            and starting with a concept.
        """
        self.stream = stream
        self.drifts = []
        self._root = None
        self._moa_backed = None

        if CLI is None:
            self._validate_stream_definition(self.stream)

            components = list(self.stream)
            self.range_form = self._uses_range_form(components)
            if self.range_form:
                concepts, self.drifts = self._resolve_range_form(components)
            else:
                concepts = components[0::2]
                self.drifts = components[1::2]
                self._check_drifts_are_placed(self.drifts)

            # Build the same shape MOA would nest: each drift mixes everything
            # before it with the concept that follows.
            root = _ConceptNode(concepts[0])
            for drift, concept in zip(self.drifts, concepts[1:]):
                root = _Transition(root, _ConceptNode(concept), drift)
            self._root = root
            self._schema = schema or concepts[0].get_schema()
            self._check_concepts_agree(concepts)
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

            self._moa_backed = MOAStream(schema=schema, CLI=CLI, moa_stream=moa_stream)
            self._schema = self._moa_backed.get_schema()

        self._CLI = CLI

    @staticmethod
    def _uses_range_form(components):
        """Decide which of the two forms the definition uses, and reject a mix.

        The two forms cannot be combined. A range definition does not say where
        its drifts land, so a stray ``position`` alongside it would describe a
        location the rest of the definition contradicts.
        """
        wrapped = [isinstance(c, Concept) for c in components[0::2]]
        if all(wrapped):
            return True
        if not any(wrapped):
            return False
        raise ValueError(
            "DriftStream concepts must either all be wrapped in ``Concept`` "
            "(the range form, giving each concept a length) or none of them "
            "(the position form, giving each drift a position). "
            f"Got {sum(wrapped)} wrapped out of {len(wrapped)} concepts."
        )

    @staticmethod
    def _check_drifts_are_placed(drifts):
        """Every drift needs a position when the definition uses positions."""
        for i, drift in enumerate(drifts):
            if drift.position is None:
                raise ValueError(
                    f"{type(drift).__name__} at drift {i} has no position. "
                    "Give it one, or use the range form by wrapping every "
                    "concept in ``Concept(stream, num_instances=...)``."
                )

    @staticmethod
    def _resolve_range_form(components):
        """Turn concept and drift lengths into concrete drift positions.

        Each component contributes its ``num_instances`` in order, so a
        definition reads as a timeline: a concept runs for its length, an
        abrupt drift switches at the point it is reached, and a gradual drift
        spans its own stretch of the stream centred on that point.
        """
        concepts = []
        drifts = []
        cursor = 0
        for component in components:
            if isinstance(component, Concept):
                concepts.append(component.stream)
                cursor += component.num_instances
                continue

            if component.position is not None or (
                isinstance(component, GradualDrift) and component.num_instances is None
            ):
                raise ValueError(
                    f"{type(component).__name__} cannot carry a position in a "
                    "range definition -- the concept lengths already determine "
                    "where it lands. Use ``AbruptDrift()`` or "
                    "``GradualDrift(num_instances=...)``."
                )

            if isinstance(component, GradualDrift):
                width = component.num_instances
                drifts.append(
                    GradualDrift(
                        position=cursor + width // 2,
                        width=width,
                        random_seed=component.random_seed,
                    )
                )
                cursor += width
            else:
                drifts.append(
                    AbruptDrift(position=cursor, random_seed=component.random_seed)
                )
        return concepts, drifts

    @staticmethod
    def _check_concepts_agree(concepts):
        """Reject concepts that do not describe the same learning problem.

        Instances from every concept are handed to the same learner and scored
        by the same evaluator, using the schema of the first concept. A concept
        that disagrees produces instances the rest of the pipeline
        misinterprets rather than an error -- a classification concept drifting
        into a regression one keeps reporting a classification schema while
        yielding ``RegressionInstance``. MOA used to reject the mismatch for
        us; it has to be checked here now.
        """

        def describe(schema):
            task = "regression" if schema.is_regression() else "classification"
            return {
                "task": task,
                "attributes": schema.get_num_attributes(),
                "numeric attributes": schema.get_num_numeric_attributes(),
                "nominal attributes": schema.get_num_nominal_attributes(),
                "nominal attribute values": schema.get_nominal_attributes(),
                # Only meaningful for classification; regression schemas report
                # nothing useful here.
                "classes": schema.get_num_classes()
                if task == "classification"
                else None,
                "labels": (
                    schema.get_label_values() if task == "classification" else None
                ),
            }

        reference = describe(concepts[0].get_schema())
        for i, concept in enumerate(concepts[1:], start=1):
            other = describe(concept.get_schema())
            differences = [
                f"{name} ({reference[name]!r} vs {other[name]!r})"
                for name in reference
                if reference[name] != other[name]
            ]
            if differences:
                raise ValueError(
                    "All concepts in a DriftStream must describe the same "
                    "learning problem, because instances from each are given to "
                    f"the same learner. Concept 0 and concept {i} differ in "
                    + "; ".join(differences)
                    + "."
                )

    @property
    def _source(self):
        return self._moa_backed if self._root is None else self._root

    def next_instance(self):
        return self._source.next_instance()

    def has_more_instances(self):
        return self._source.has_more_instances()

    def get_schema(self):
        return self._schema

    def restart(self):
        if self._root is not None:
            self._root.restart()
        else:
            self._moa_backed.restart()

    def get_moa_stream(self):
        """Return the backing MOA stream, if there is one.

        A Python-composed ``DriftStream`` has no single MOA object behind it.
        Use :func:`to_moa_stream` to build one.
        """
        if self._moa_backed is not None:
            return self._moa_backed.get_moa_stream()
        return None

    @property
    def moa_stream(self):
        """The backing MOA stream object, or ``None`` when composed in Python.

        Kept so a ``DriftStream`` built from a MOA CLI still works with code
        that reaches for the attribute directly, such as the optimised
        evaluation loops.
        """
        if self._moa_backed is not None:
            return self._moa_backed.moa_stream
        return None

    def to_moa_stream(self):
        """Build the equivalent MOA ``ConceptDriftStream``.

        Provided for interoperability with MOA code and for inspecting the
        generated CLI. Requires every concept to be MOA-backed, which is the
        restriction Python composition exists to remove -- so this raises when
        the stream contains a concept MOA cannot represent.

        :return: A :class:`~capymoa.stream.MOAStream` wrapping the nested
            ``ConceptDriftStream``.
        """
        if self._moa_backed is not None:
            return self._moa_backed
        if self.stream is None:
            raise ValueError("This DriftStream has no definition to convert.")

        not_moa = [
            type(component).__name__
            for component in self.stream[0::2]
            if not isinstance(component, MOAStream)
        ]
        if not_moa:
            raise ValueError(
                "to_moa_stream() needs every concept to be MOA-backed, but this "
                f"stream uses {', '.join(sorted(set(not_moa)))}. MOA cannot "
                "represent Python-native concepts."
            )

        stream1 = self.stream[0]
        CLI = ""
        for drift, concept in zip(self.stream[1::2], self.stream[2::2]):
            CLI = f" -s {cli_str_stream(stream1.moa_stream)} "
            CLI += (
                f" -d {cli_str_stream(concept.moa_stream)} -w {drift.width} -p "
                f"{drift.position} -r {drift.random_seed}"
            )
            # got to remove package name from streams.ConceptDriftStream
            CLI = CLI.replace("streams.", "")
            stream1 = MOAStream(moa_stream=MOA_ConceptDriftStream(), CLI=CLI)
        return stream1

    @staticmethod
    def _validate_stream_definition(stream):
        """Check the concept/drift list before anything is composed.

        The definition has to alternate concept, drift, concept, ... starting
        and ending with a concept. Validating up front matters because the
        builder appends each ``Drift`` to ``self.drifts`` as it sees it, before
        it knows whether a concept follows: a malformed list would otherwise
        report drifts through :func:`get_num_drifts` that were never composed
        into the stream.
        """
        if not stream:
            raise ValueError(
                "DriftStream needs a non-empty list of concepts and drifts, "
                "e.g. [concept, drift, concept]."
            )

        def kind(component):
            if isinstance(component, Drift):
                return "drift"
            if isinstance(component, (Stream, Concept)):
                return "concept"
            return None

        for i, component in enumerate(stream):
            actual = kind(component)
            if actual is None:
                raise ValueError(
                    f"DriftStream cannot use {type(component).__name__} as a "
                    "component. Concepts must be ``Stream`` objects, or "
                    "``Concept`` when giving lengths, and drifts must be "
                    "``Drift`` objects."
                )
            expected = "concept" if i % 2 == 0 else "drift"
            if actual != expected:
                raise ValueError(
                    "DriftStream expects concepts and drifts to alternate, "
                    "starting and ending with a concept, e.g. "
                    "[concept, drift, concept]. Position "
                    f"{i} is a {actual} where a {expected} was expected."
                )

        if kind(stream[-1]) == "drift":
            raise ValueError(
                "DriftStream must end with a concept, not a drift: a trailing "
                f"{type(stream[-1]).__name__} has no concept to drift into. "
                f"Got {len(stream)} components ending with a drift."
            )

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


class Concept:
    """A concept and the number of instances it contributes to the stream.

    Used by the *range* form of :class:`DriftStream`, which specifies how long
    each concept lasts instead of where each drift lands:

    >>> from capymoa.stream.drift import Concept, DriftStream, AbruptDrift
    >>> from capymoa.stream.generator import SEA
    >>> stream = DriftStream(stream=[
    ...     Concept(SEA(function=1), num_instances=1000),
    ...     AbruptDrift(),
    ...     Concept(SEA(function=3), num_instances=500),
    ... ])
    >>> print(stream.get_drifts()[0])
    AbruptDrift(position=1000)

    The wrapper exists because :class:`~capymoa.stream.Stream` has no notion of
    a length: MOA generators are unbounded, and giving every stream a length
    to serve this one API would not survive contact with them.

    .. note::

        ``num_instances`` places the drifts; it does not ration instances.
        Around an :class:`AbruptDrift` it is exact, because the switch is a
        step. Around a :class:`GradualDrift` the two concepts overlap, so both
        contribute while the transition runs, and with the default sigmoid the
        older concept keeps appearing -- with diminishing probability -- past
        the nominal end of the window. A concept declared as 500 instances may
        therefore be drawn from rather more often than 500 times. See
        :class:`GradualDrift` for the shape of the ramp.
    """

    def __init__(self, stream, num_instances: int):
        """
        :param stream: The concept to draw instances from.
        :param num_instances: How many instances this concept contributes.
        """
        if num_instances is None or num_instances <= 0:
            raise ValueError(
                f"Concept needs a positive ``num_instances``, got {num_instances!r}."
            )
        self.stream = stream
        self.num_instances = num_instances

    def get_schema(self):
        return self.stream.get_schema()

    def __str__(self):
        return f"Concept({self.stream}, num_instances={self.num_instances})"


class Drift:
    """Represents a concept drift in a DriftStream. See 2.7.1 Concept drift framework in [1]_.

    .. [1] Bifet, Albert, et al. "Data stream mining: a practical approach." COSI (2011).
    """

    def __init__(self, position, width=0, *, random_seed=1):
        """Construct a drift in a DriftStream.

        :param position: The location of the drift in terms of the number of instances processed prior to it occurring.
        :param width: The size of the window of change. A width of 0 or 1 corresponds to an abrupt drift.
        :param random_seed: Seed for random number generation, defaults to 1.
            Keyword-only, so a leftover positional argument from the removed
            ``alpha`` parameter fails at the call site rather than silently
            becoming the seed.
        """
        self.width = width
        self.position = position
        self.random_seed = random_seed

    def __str__(self):
        drift_kind = "GradualDrift"
        if self.width == 0 or self.width == 1:
            drift_kind = "AbruptDrift"
        attributes = [
            f"position={self.position}",
            f"width={self.width}" if self.width not in [0, 1] else None,
            f"random_seed={self.random_seed}" if self.random_seed != 1 else None,
        ]
        non_default_attributes = [attr for attr in attributes if attr is not None]
        return f"{drift_kind}({', '.join(non_default_attributes)})"


class GradualDrift(Drift):
    """A drift where two concepts overlap over a window of instances.

    The location can be given in either of two mutually exclusive ways, which
    describe the same drift:

    >>> from capymoa.stream.drift import GradualDrift
    >>> print(GradualDrift(position=100, width=10))
    GradualDrift(position=100, start=95, end=105, width=10)
    >>> print(GradualDrift(start=95, end=105))
    GradualDrift(position=100, start=95, end=105, width=10)

    ``start`` and ``end`` mark where the transition is centred, not where the
    concepts stop overlapping. The default ramp is the logistic MOA uses,
    ``1 / (1 + exp(-4 (n - position) / width))``, which reaches 0.12 at
    ``start`` and 0.88 at ``end`` and approaches 0 and 1 only asymptotically.
    So roughly 12% of instances at the nominal end still come from the older
    concept, and a few continue to appear well beyond it. A ramp that
    saturates exactly at the window edges is a different function, not a
    different width.

    A third form gives the drift a length and lets :class:`DriftStream` work
    out where it lands from the concepts around it -- see :class:`Concept`:

    >>> unplaced = GradualDrift(num_instances=500)
    >>> unplaced.width, unplaced.position
    (500, None)

    Supplying neither style, or only half of one, is an error -- rather than
    building a drift with no location:

    >>> GradualDrift(position=100)
    Traceback (most recent call last):
        ...
    ValueError: GradualDrift needs exactly one of ``position`` and ``width``, ``start`` and ``end``, or ``num_instances``, to locate the drift. Got position=100.

    >>> GradualDrift(position=100, start=95)
    Traceback (most recent call last):
        ...
    ValueError: GradualDrift needs exactly one of ``position`` and ``width``, ``start`` and ``end``, or ``num_instances``, to locate the drift. Got position=100, start=95.
    """

    def __init__(
        self,
        position=None,
        width=None,
        start=None,
        end=None,
        *,
        num_instances=None,
        random_seed=1,
    ):
        self.__init_args_kwargs__ = copy.copy(
            locals()
        )  # save init args for recreation. not a deep copy to avoid unnecessary use of memory

        # Python has no function overloading, so the location of the drift can be
        # given in one of several mutually exclusive ways. Validate that exactly
        # one complete style was supplied *before* assigning anything, so an
        # invalid call raises here rather than leaving a half-built object that
        # fails later somewhere unrelated.
        styles = {
            "position and width": (position is not None, width is not None),
            "start and end": (start is not None, end is not None),
            # Range form: the drift spans this many instances, and DriftStream
            # works out where it lands from the concepts around it.
            "num_instances": (num_instances is not None,),
        }
        complete = [name for name, given in styles.items() if all(given)]
        partial = [
            name for name, given in styles.items() if any(given) and not all(given)
        ]

        if len(complete) != 1 or partial:
            supplied = ", ".join(
                f"{name}={value!r}"
                for name, value in (
                    ("position", position),
                    ("width", width),
                    ("start", start),
                    ("end", end),
                    ("num_instances", num_instances),
                )
                if value is not None
            )
            raise ValueError(
                "GradualDrift needs exactly one of "
                "``position`` and ``width``, ``start`` and ``end``, or "
                "``num_instances``, to locate the drift. "
                f"Got {supplied if supplied else 'no arguments'}."
            )

        self.num_instances = num_instances
        if complete == ["num_instances"]:
            # Unplaced: DriftStream resolves it against the surrounding
            # concepts, because the position is not knowable here.
            self.width = num_instances
            self.position = None
            self.start = None
            self.end = None
            self.random_seed = random_seed
            super().__init__(
                position=None, random_seed=random_seed, width=num_instances
            )
            return

        if complete == ["position and width"]:
            self.width = width
            self.position = position
            self.start = int(position - width / 2)
            self.end = int(position + width / 2)
        else:
            self.start = start
            self.end = end
            self.width = end - start
            self.position = int((start + end) / 2)

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
            f"random_seed={self.random_seed}" if self.random_seed != 1 else None,
        ]
        non_default_attributes = [attr for attr in attributes if attr is not None]
        return f"GradualDrift({', '.join(non_default_attributes)})"


class AbruptDrift(Drift):
    """An instantaneous change of concept at ``position``.

    >>> from capymoa.stream.drift import AbruptDrift
    >>> print(AbruptDrift(position=5000))
    AbruptDrift(position=5000)

    ``position`` may be omitted only in the *range* form of
    :class:`DriftStream`, where the lengths of the surrounding
    :class:`Concept` objects decide where the drift lands:

    >>> print(AbruptDrift())
    AbruptDrift()

    A :class:`DriftStream` built from positions rejects a drift without one,
    so an omitted position cannot quietly become a drift at instance zero.
    """

    def __init__(self, position: int = None, random_seed: int = 1):
        self.__init_args_kwargs__ = copy.copy(
            locals()
        )  # save init args for recreation. not a deep copy to avoid unnecessary use of memory

        self.position = position
        self.random_seed = random_seed

        super().__init__(position=position, random_seed=random_seed)

    def __str__(self):
        attributes = [
            f"position={self.position}" if self.position is not None else None,
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
    if not isinstance(transition_type_template, (AbruptDrift, GradualDrift)):
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
    if isinstance(transition_type_template, GradualDrift):
        original_drift_args["position"] = transition_type_template.position
        original_drift_args["width"] = transition_type_template.width
        original_drift_args["start"] = None
        original_drift_args["end"] = None

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
