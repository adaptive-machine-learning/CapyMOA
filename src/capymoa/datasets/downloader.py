import gzip
import shutil
from abc import ABC, abstractmethod
from os import environ
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Optional
import shutil

import wget
from moa.streams import ArffFileStream

from capymoa.stream.stream import Stream

CAPYMOA_DATASETS_DIR = environ.get("CAPYMOA_DATASETS_DIR", "data")
"""A default directory to store datasets in. Defaults to `./data` when the
environment variable `CAPYMOA_DATASETS_DIR` is not set.
"""


class DownloadableDataset(ABC, Stream):
    filename: str = None
    """Name of the dataset in the capymoa dataset directory"""

    def __init__(
        self,
        directory: str = CAPYMOA_DATASETS_DIR,
        auto_download: bool = True,
        CLI: Optional[str] = None,
        schema: Optional[str] = None,
    ):
        assert self.filename is not None, "Filename must be set in subclass"
        self._path = self._resolve_dataset(
            auto_download,
            Path(directory).resolve(),
        )
        moa_stream = self.to_stream(self._path)
        super().__init__(schema=schema, CLI=CLI, moa_stream=moa_stream)

    def _resolve_dataset(self, auto_download: bool, directory: Path):
        directory.mkdir(parents=True, exist_ok=True)
        stream = directory / self.filename

        if not stream.exists():
            if auto_download:
                with TemporaryDirectory() as working_directory:
                    working_directory = Path(working_directory)
                    stream_archive = self.download(working_directory)
                    tmp_stream = self.extract(stream_archive)
                    stream = shutil.move(tmp_stream, stream)
            else:
                raise FileNotFoundError(
                    f"Dataset {self.filename} not found in {directory}"
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


class DownloadARFFGzip(DownloadableDataset):
    filename = None
    remote_url = None

    def download(self, working_directory: Path) -> Path:
        assert self.remote_url is not None, "Remote URL must be set in subclass"

        print(f"Downloading {self.filename}")

        archive_filename = self.filename + ".gz"
        save_path = working_directory / archive_filename
        remote_path = self.remote_url + archive_filename
        path = wget.download(remote_path, save_path.as_posix())
        return Path(path)

    def extract(self, stream_archive: Path) -> Path:
        # Remove the .gz extension
        assert stream_archive.suffix == ".gz"
        stream: Path = stream_archive.with_suffix("")

        # Extract the archive
        with gzip.open(stream_archive, "rb") as f_in:
            with open(stream, "wb") as f_out:
                shutil.copyfileobj(f_in, f_out)
        return stream

    def to_stream(self, stream: Path) -> Any:
        return ArffFileStream(stream.as_posix(), -1)
