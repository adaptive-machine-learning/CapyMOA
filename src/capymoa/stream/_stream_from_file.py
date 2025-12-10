from pathlib import Path
from typing import Union, Literal
from capymoa.stream._stream import ARFFStream, Stream
from capymoa.stream._csv_stream import CSVStream
import pandas as pd


def stream_from_file(
    path_to_csv_or_arff: Union[str, Path],
    dataset_name: str = "NoName",
    class_index: int = -1,
    target_type: Literal["numeric", "categorical"] | None = None,
) -> Stream:
    """Create a datastream from a csv or arff file.

    >>> from capymoa.stream import stream_from_file
    >>> stream = stream_from_file(
    ...     "data/electricity_tiny.csv",
    ...     dataset_name="Electricity",
    ...     target_type="categorical"
    ... )
    >>> stream.next_instance()
    LabeledInstance(
        Schema(Electricity),
        x=[0.    0.056 0.439 0.003 0.423 0.415],
        y_index=1,
        y_label='1'
    )
    >>> stream.next_instance().x
    array([0.021277, 0.051699, 0.415055, 0.003467, 0.422915, 0.414912])

    ..  seealso::

        * :class:`~capymoa.stream.CSVStream`
        * :class:`~capymoa.stream.ARFFStream`
        * :class:`~capymoa.stream.NumpyStream`
        * :class:`~capymoa.stream.TorchClassifyStream`
        * :class:`~capymoa.stream.Stream`

    **CSV File Considerations:**

    * Assumes a **header row** with attribute names.
    * Supports only **numeric** and **categorical** attributes.
    * String columns are automatically converted to **categorical** attributes. Convert
      them to numeric beforehand if needed.
    * The whole CSV file is **read into memory**. For very large files, use
      :class:`~capymoa.stream.CSVStream` directly for streaming from disk.
    * **Default Target:** The last column (`class_index=-1`) is assumed to be the target
      variable.
    * **Missing Values:** Represented by ``?``.

    **ARFF File Considerations:**

    Reads the Attribute-Relation File Format (`ARFF
    <https://waikato.github.io/weka-wiki/formats_and_processing/arff_stable/>`_)
    commonly used in MOA and WEKA.

    * Supports only ``NUMERIC`` and ``NOMINAL`` attributes.
    * **Missing Values:** Represented by ``?``.
    * **Default Target:** The last attribute is assumed to be the target variable.



    :param path_to_csv_or_arff: A file path to a CSV or ARFF file.
    :param dataset_name: A descriptive name given to the dataset, defaults to "NoName"
    :param class_index: The index of the column containing the class label. By default,
        the algorithm assumes that the class label is located in the column specified by
        this index. However, if the class label is located in a different column, you
        can specify its index using this parameter.
    :param target_type: When working with a CSV file, this parameter allows the user to
        specify the target values in the data to be interpreted as categorical or
        numeric. Defaults to None to detect automatically.
    """
    filename = Path(path_to_csv_or_arff)
    if not filename.exists():
        raise FileNotFoundError(f"No such file or directory: '{filename}'")
    if filename.is_dir():
        raise IsADirectoryError(f"Is a directory: '{filename}'")

    if filename.suffix == ".arff":
        return ARFFStream(path=filename.as_posix(), class_index=class_index)
    elif filename.suffix == ".csv":
        # Read CSV file
        df = pd.read_csv(filename, na_values=["?", "NA", "NaN"])
        target = df.columns[class_index]
        categories = {}

        # set target column type
        if target_type == "categorical":
            df[target] = df[target].astype("category")
            categories[target] = list(df[target].cat.categories)

        # convert string columns into categories
        for col in df.select_dtypes(include=["object"]).columns:
            df[col] = df[col].astype("category")
            categories[col] = list(df[col].cat.categories)

        # Remove ? from categories if present
        for col, cats in categories.items():
            if "?" in cats:
                cats.remove("?")

        return CSVStream(
            file=filename,
            categories=categories,
            target=target,
            name=dataset_name,
            length=len(df),
        )
    else:
        raise ValueError(
            f"Unsupported file type: expected '.arff' or '.csv', but got '{filename.suffix}'"
        )
