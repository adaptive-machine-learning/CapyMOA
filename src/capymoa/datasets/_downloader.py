from abc import ABC, abstractmethod
from pathlib import Path
from typing import Literal, Optional, Union

from capymoa.stream._stream import Schema
from moa.streams import InstanceStream as _InstanceStream

from capymoa.stream import Stream, stream_from_file
from capymoa.datasets._utils import (
    download_unpacked,
    get_download_dir,
    infer_unpacked_path,
)
from capymoa.datasets._source_list import SOURCE_LIST


FileType = Literal["arff", "csv"]


def _resolve_dataset_url(dataset_name: str, file_type: FileType) -> str:
    source = SOURCE_LIST[dataset_name]

    if file_type == "arff":
        return source.arff
    if file_type == "csv":
        if source.csv is None:
            raise ValueError(f"Dataset {dataset_name} does not provide a CSV download.")
        return source.csv

    raise ValueError(f"Unsupported dataset file type: {file_type}")


class _DownloadableDataset(ABC):
    _length: int
    """Number of instances in the dataset"""
    _url: str
    """URL to a file to download."""

    def __init__(
        self,
        directory: Union[str, Path] = get_download_dir(),
        auto_download: bool = True,
        file_type: FileType = "arff",
    ):
        self.file_type: FileType = file_type
        self._url = _resolve_dataset_url(type(self).__name__, file_type)
        self.path = infer_unpacked_path(self._url, directory)
        if not self.path.exists():
            if auto_download:
                download_unpacked(self._url, directory)
            else:
                raise FileNotFoundError(
                    f"Dataset {self.path.name} not found in {directory}. "
                    "Try downloading it with `auto_download=True`."
                )

    @classmethod
    @abstractmethod
    def to_stream(cls, path: Path) -> Stream:
        """Convert the downloaded and unpacked dataset into a datastream."""

    def __len__(self) -> int:
        return self._length

    def __str__(self) -> str:
        return type(self).__name__


class _DownloadableARFF(_DownloadableDataset, Stream):
    schema: Schema
    stream: Stream
    moa_stream: Optional[_InstanceStream]
    _target_type: Literal["numeric", "categorical"] | None = None

    def __init__(
        self,
        directory: Union[str, Path] = get_download_dir(),
        auto_download: bool = True,
        file_type: FileType = "arff",
    ):
        """Setup a stream from a dataset file and optionally download it if missing.

        :param directory: Where downloads are stored.
            Defaults to :func:`capymoa.datasets.get_download_dir`.
        :param auto_download: Download the dataset if it is missing.
        :param file_type: Download either the ``"arff"`` or ``"csv"`` dataset asset.
        """
        _DownloadableDataset.__init__(self, directory, auto_download, file_type)
        self.stream = self.to_stream(self.path)
        self.schema = self.stream.get_schema()
        self.moa_stream = self.stream.get_moa_stream()

    @classmethod
    def to_stream(cls, path: Path) -> Stream:
        return stream_from_file(
            path,
            dataset_name=cls.__name__,
            target_type=cls._target_type,
        )

    def has_more_instances(self) -> bool:
        return self.stream.has_more_instances()

    def next_instance(self):
        return self.stream.next_instance()

    def get_schema(self) -> Schema:
        return self.schema

    def get_moa_stream(self) -> Optional[_InstanceStream]:
        return self.moa_stream

    def restart(self):
        self.stream.restart()
