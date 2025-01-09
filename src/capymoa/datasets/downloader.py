import shutil
from abc import ABC, abstractmethod
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Optional

import wget
from moa.streams import ArffFileStream

from capymoa.stream import MOAStream
from capymoa.datasets._utils import extract, get_download_dir
import os


class DownloadableDataset(MOAStream, ABC):
    _filename: str = None
    """Name of the dataset in the capymoa dataset directory"""
    _length: int
    """Number of instances in the dataset"""

    def __init__(
        self,
        directory: str = get_download_dir(),
        auto_download: bool = True,
        CLI: Optional[str] = None,
        schema: Optional[str] = None,
    ):
        assert self._filename is not None, "Filename must be set in subclass"
        self._path = self._resolve_dataset(
            auto_download,
            Path(directory).resolve(),
        )
        moa_stream = self.to_stream(self._path)
        super().__init__(schema=schema, CLI=CLI, moa_stream=moa_stream)

    def _resolve_dataset(self, auto_download: bool, directory: Path):
        directory.mkdir(parents=True, exist_ok=True)
        stream = directory / self._filename

        if not stream.exists():
            if auto_download:
                with TemporaryDirectory() as working_directory:
                    working_directory = Path(working_directory)
                    stream_archive = self.download(working_directory)
                    tmp_stream = self.extract(stream_archive)
                    stream = shutil.move(tmp_stream, stream)
            else:
                raise FileNotFoundError(
                    f"Dataset {self._filename} not found in {directory}"
                )

        return stream

    def get_path(self):
        return self._path

    @abstractmethod
    def download(self, working_directory: Path) -> Path:
        """Download the dataset and return the path to the downloaded dataset
        within the working directory.

        :param working_directory: The directory to download the dataset to.
        :return: The path to the downloaded dataset within the working directory.
        """
        pass

    @abstractmethod
    def extract(self, stream_archive: Path) -> Path:
        """Extract the dataset from the archive and return the path to the
        extracted dataset.

        :param stream_archive: The path to the archive containing the dataset.
        :return: The path to the extracted dataset.
        """
        pass

    @abstractmethod
    def to_stream(self, stream: Path):
        """Convert the dataset to a MOA stream.

        :param stream: The path to the dataset.
        :return: A MOA stream.
        """
        pass

    def __len__(self) -> int:
        return self._length

    def __str__(self) -> str:
        return type(self).__name__


class DownloadARFFGzip(DownloadableDataset):
    _remote_url = None

    def download(self, working_directory: Path) -> Path:
        assert self._remote_url is not None, "Remote URL must be set in subclass"

        print(f"Downloading {self._filename}")
        # wget creates temporary files in the current working directory. We need to
        # change the working directory to avoid cluttering the current directory.
        wd = os.getcwd()
        os.chdir(working_directory)
        path = wget.download(self._remote_url, working_directory.as_posix())
        os.chdir(wd)
        return Path(path)

    def extract(self, stream_archive: Path) -> Path:
        return extract(stream_archive)

    def to_stream(self, stream: Path) -> Any:
        return ArffFileStream(stream.as_posix(), -1)
