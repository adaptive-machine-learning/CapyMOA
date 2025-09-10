from abc import ABC, abstractmethod
from pathlib import Path
from typing import Union

from capymoa.stream._stream import Schema
from moa.streams import ArffFileStream, InstanceStream

from capymoa.stream import MOAStream
from capymoa.datasets._utils import (
    download_unpacked,
    get_download_dir,
    infer_unpacked_path,
)


class _DownloadableDataset(ABC):
    _length: int
    """Number of instances in the dataset"""
    _url: str
    """URL to a file to download."""

    def __init__(
        self,
        directory: Union[str, Path] = get_download_dir(),
        auto_download: bool = True,
    ):
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
    def to_stream(cls, path: Path) -> InstanceStream:
        """Convert the downloaded and unpacked dataset into a datastream."""

    def __len__(self) -> int:
        return self._length

    def __str__(self) -> str:
        return type(self).__name__


class _DownloadableARFF(_DownloadableDataset, MOAStream):
    schema: Schema

    def __init__(
        self,
        directory: Union[str, Path] = get_download_dir(),
        auto_download: bool = True,
    ):
        """Setup a stream from an ARFF file and optionally download it if missing.

        :param directory: Where downloads are stored.
            Defaults to :func:`capymoa.datasets.get_download_dir`.
        :param auto_download: Download the dataset if it is missing.
        """
        _DownloadableDataset.__init__(self, directory, auto_download)
        MOAStream.__init__(self, self.to_stream(self.path))

    @classmethod
    def to_stream(cls, path: Path) -> InstanceStream:
        return ArffFileStream(path.as_posix(), -1)
