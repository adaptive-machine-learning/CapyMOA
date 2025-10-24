from capymoa._pickle import (
    JPickler as DeprecatedJPickler,
    JUnpickler as DeprecatedJUnpickler,
)
from jpype.pickle import JPickler, JUnpickler
from deprecated import deprecated
from jpype import JException
from typing import BinaryIO, TextIO
from pathlib import Path
from io import RawIOBase, BufferedIOBase
from capymoa.stream._stream import Stream
import tqdm


# TODO: Remove this and capymoa._pickle in a future release
@deprecated(version="v0.8.2", reason="Use ``save_model(...)`` instead.")
def legacy_save_model(model, filename):
    """Save a model to a file.

    Use :func:`save_model` if possible.

    :param model: The model to save.
    :param filename: The file to save the model to.
    """

    with open(filename, "wb") as fd:
        DeprecatedJPickler(fd).dump(model)


# TODO: Remove this and capymoa._pickle in a future release
@deprecated(version="v0.8.2", reason="Use ``load_model(...)`` instead.")
def legacy_load_model(filename):
    """Load a model from a file.

    Use :func:`load_model` if possible.

    :param filename: The file to load the model from.
    """

    with open(filename, "rb") as fd:
        return DeprecatedJUnpickler(fd).load()


def save_model(model: object, file: BinaryIO) -> None:
    """Save a model to a jpype pickle file.

    >>> from capymoa.classifier import AdaptiveRandomForestClassifier
    >>> from capymoa.datasets import ElectricityTiny
    >>> from tempfile import TemporaryFile
    >>> stream = ElectricityTiny()
    >>> learner = AdaptiveRandomForestClassifier(schema=stream.get_schema())
    >>> with TemporaryFile() as fd:
    ...     save_model(learner, fd)

    See https://jpype.readthedocs.io/en/latest/api.html#jpype-pickle-module for
    more information.

    :param model: A python object optionally containing Java objects.
    :param file: The file-like object to save the model to.
    """
    if not file.writable():
        raise ValueError("File must be writable.")
    JPickler(file).dump(model)


def load_model(file: BinaryIO) -> object:
    """Load a model from a jpype pickle file.

    If you are trying to load a model saved with a version of CapyMOA < 0.8.2,
    use :func:`legacy_load_model` and :func:`save_model` to reformat the model.

    See also: :func:`save_model`.

    :param file: The file-like object to load the model from.
    :return: The loaded model.
    """
    if not isinstance(file, (RawIOBase, BufferedIOBase)):
        raise ValueError("File must be opened in binary mode.")
    if not file.readable():
        raise ValueError("File must be readable.")
    try:
        return JUnpickler(file).load()
    except JException as e:
        raise RuntimeError(
            "Exception loading model.\n"
            "If you are trying to load a model saved with a version of CapyMOA < 0.8.2, "
            "use `legacy_load_model` and `save_model` to reformat the model."
        ) from e


def save_stream_arff(file: TextIO | Path | str, stream: Stream) -> None:
    """Save a CapyMOA stream to an ARFF file.

    Usage for classification datastream:

    >>> from capymoa.stream import NumpyStream
    >>> import numpy as np
    >>> from io import StringIO
    >>>
    >>> stream = NumpyStream(
    ...     X=np.array([[0, 1], [1, 0], [0, 0]]),
    ...     y=np.array([0, 1, 0]),
    ...     dataset_name="SimpleDataset",
    ...     target_type="categorical"
    ... )
    >>> fd = StringIO() # You can 'open' a real file instead
    >>> save_stream_arff(fd, stream)
    >>> print(fd.getvalue())
    @relation SimpleDataset
    <BLANKLINE>
    @attribute attrib_0 numeric
    @attribute attrib_1 numeric
    @attribute class {0,1}
    <BLANKLINE>
    @data
    0.0,1.0,0,
    1.0,0.0,1,
    0.0,0.0,0,
    <BLANKLINE>

    Usage for regression datastream:

    >>> stream = NumpyStream(
    ...     X=np.array([[0, 1], [1, 0], [0, 0]]),
    ...     y=np.array([0, 1, 0]),
    ...     dataset_name="SimpleDataset",
    ...     target_type="numeric"
    ... )
    >>> fd = StringIO() # You can 'open' a real file instead
    >>> save_stream_arff(fd, stream)
    >>> print(fd.getvalue())
    @relation SimpleDataset
    <BLANKLINE>
    @attribute attrib_0 numeric
    @attribute attrib_1 numeric
    @attribute target numeric
    <BLANKLINE>
    @data
    0.0,1.0,0.0,
    1.0,0.0,1.0,
    0.0,0.0,0.0,
    <BLANKLINE>


    :param file: A file-like object or path to write the ARFF to.
    :param stream: The stream to save.
    """
    if isinstance(file, (Path, str)):
        with open(file, "w") as fd:
            return save_stream_arff(fd, stream)
    else:
        moa_header = stream.get_schema().get_moa_header()
        file.write(str(moa_header))
        for instance in tqdm.tqdm(stream, desc="Saving to ARFF"):
            file.write(str(instance.java_instance.toString()) + "\n")
