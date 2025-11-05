from pathlib import Path
from capymoa.instance import LabeledInstance, RegressionInstance
from capymoa.stream._stream import Stream, Schema, _AnyInstance
from typing import Dict, TextIO, Sequence, Optional
from typing_extensions import override
import csv


class CSVStream(Stream[_AnyInstance]):
    """Create a CapyMOA datastream from a CSV file.

    * The CSV file **must** have a header row with feature names.

    * Integers or strings can specify nominal features.

    When 'categories' are provided for the target attribute, then the stream returns
    :class:`~capymoa.instance.LabeledInstance` objects.

    >>> from io import StringIO
    >>> from capymoa.stream import CSVStream
    >>> csv_content = '''feature1,feature2,target
    ... 1,A,yes
    ... 2,B,no
    ... 3,0,0
    ... 5,1,1
    ... '''
    >>> csv_file = StringIO(csv_content)
    >>> stream = CSVStream(
    ...     file=csv_file,
    ...     target="target",
    ...     categories={"target": ["yes", "no"], "feature2": ["A", "B"]},
    ...     name="TestStream"
    ... )
    >>> for instance in stream:
    ...     print(instance)
    LabeledInstance(
        Schema(TestStream),
        x=[1. 0.],
        y_index=0,
        y_label='yes'
    )
    LabeledInstance(
        Schema(TestStream),
        x=[2. 1.],
        y_index=1,
        y_label='no'
    )
    LabeledInstance(
        Schema(TestStream),
        x=[3. 0.],
        y_index=0,
        y_label='yes'
    )
    LabeledInstance(
        Schema(TestStream),
        x=[5. 1.],
        y_index=1,
        y_label='no'
    )

    When no categories are provided for the target attribute, then the stream returns
    :class:`~capymoa.instance.RegressionInstance` objects.

    >>> csv_content = '''target,feature1,feature2
    ... 0.0,A,1
    ... 0.5,B,2
    ... 1.5,0,3
    ... 2.0,1,4
    ... '''
    >>> csv_file = StringIO(csv_content)
    >>> stream = CSVStream(
    ...     file=csv_file,
    ...     target="target",
    ...     categories={"feature1": ["A", "B"]},
    ...     name="TestStream"
    ... )
    >>> for instance in stream:
    ...     print(instance)
    RegressionInstance(
        Schema(TestStream),
        x=[0. 1.],
        y_value=0.0
    )
    RegressionInstance(
        Schema(TestStream),
        x=[1. 2.],
        y_value=0.5
    )
    RegressionInstance(
        Schema(TestStream),
        x=[0. 3.],
        y_value=1.5
    )
    RegressionInstance(
        Schema(TestStream),
        x=[1. 4.],
        y_value=2.0
    )

    """

    def __init__(
        self,
        file: Path | str | TextIO,
        target: str,
        categories: Optional[Dict[str, Sequence[str]]] = None,
        name: Optional[str] = None,
    ) -> None:
        """Create a CSV stream.

        :param file: A path to a CSV file or an open file-like object.
        :param target: The name of the target attribute.
        :param categories: A mapping from attribute names to their categorical values.
        :param name: An optional name for the stream. If not provided, the filename is
            used.
        """

        # Open file if a path is given
        if isinstance(file, (str, Path)):
            file = Path(file).expanduser()
            name = name or file.stem
            self._file = open(file, "r")
        else:
            self._file = file

        if categories and target in categories:
            self._instance_type = LabeledInstance
        else:
            self._instance_type = RegressionInstance

        self._reader = csv.reader(self._file, delimiter=",")
        feature_names = next(self._reader)
        feature_names = [n.strip() for n in feature_names]

        if target not in feature_names:
            raise ValueError(
                f"Target attribute '{target}' not found in CSV header. "
                f"Header contains {feature_names}."
            )

        self.schema = Schema.from_custom(
            features=feature_names,
            target=target,
            categories=categories,
            name=name or "unnamed-csv-stream",
        )
        # Buffer a single instance so that we can implement has_more_instances
        self._instance = self._next_instance()

    def _next_instance(self) -> _AnyInstance | None:
        try:
            return self._instance_type.from_csv_row(self.schema, next(self._reader))  # type: ignore
        except StopIteration:
            return None

    @override
    def get_schema(self) -> Schema:
        return self.schema

    @override
    def has_more_instances(self) -> bool:
        return self._instance is not None

    @override
    def next_instance(self) -> _AnyInstance:
        instance = self._instance
        if instance is None:
            raise StopIteration()
        self._instance = self._next_instance()
        return instance

    @override
    def restart(self) -> None:
        if self._file.seek(0) != 0:
            raise IOError("Failed to seek to the beginning of the file.")
        self._reader = csv.reader(self._file, delimiter=",")
        next(self._reader)  # Skip header
        # Buffer a single instance so that we can implement has_more_instances
        self._instance = self._next_instance()
