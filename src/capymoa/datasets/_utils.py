import gzip
from pathlib import Path
from typing import Callable, Optional, Tuple
from urllib.request import urlretrieve
import torch
from shutil import copyfileobj
from capymoa.env import capymoa_datasets_dir
import numpy as np
from urllib.parse import urlparse
from os.path import basename
from shutil import unpack_archive, get_unpack_formats, move
from tqdm import tqdm

_GZIP_SUFFIX = [".gz", ".gzip"]


def _unpacked_format(filename: str) -> Tuple[str | None, str]:
    for format, extensions, _ in get_unpack_formats():
        for ext in extensions:
            if filename.endswith(ext):
                return format, filename.removesuffix(ext)
    for ext in _GZIP_SUFFIX:
        if filename.endswith(ext):
            return "gzip", filename.removesuffix(ext)
    return None, filename


def _url_to_filename(url: str) -> str:
    return basename(urlparse(url).path)


def infer_unpacked_path(url: str, downloads: Path | str) -> Path:
    _, filename = _unpacked_format(_url_to_filename(url))
    return Path(downloads) / filename


def is_already_downloaded(url: str, downloads: Path | str) -> bool:
    return infer_unpacked_path(url, downloads).exists()


class _TqdmUpTo(tqdm):
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        return self.update(b * bsize - self.n)  # also sets self.n = b * bsize


def download_unpacked(url: str, downloads: Path | str) -> Path:
    """Download and unpack an archived/compressed file.

    * ``https://example.com/mydir.tar.gz`` -> ``/downloads/mydir/``
    * ``https://example.com/mydir.zip`` -> ``/downloads/mydir/``
    * ``https://example.com/myfile.gz`` -> ``/downloads/myfile``

    :param url: URL with a valid filename.
    :param downloads: Base directory to download to.
    :raises FileNotFoundError: If ``downloads`` does not exist.
    :raises NotADirectoryError: If ``downloads`` is not a directory.
    :raises FileExistsError: If the unpacked directory/file already exists.
    :return: The unpacked directory/file.
    """
    downloads = Path(downloads)
    format, filename = _unpacked_format(_url_to_filename(url))
    path = downloads / filename
    if not downloads.exists():
        raise FileNotFoundError()
    if not downloads.is_dir():
        raise NotADirectoryError()
    if path.exists():
        raise FileExistsError(f"File or directory '{path}' already exists.")

    with _TqdmUpTo(unit="B", unit_scale=True, unit_divisor=1024, desc=filename) as pbar:
        tmpfile, _ = urlretrieve(url, reporthook=pbar.update_to)

    # Unpack, decompress, or simply move to the path
    if format == "gzip":
        with gzip.open(tmpfile, "rb") as fin:
            with open(path, "xb") as fdst:
                copyfileobj(fin, fdst)
    elif format is not None:
        unpack_archive(tmpfile, path, format=format)
    else:
        move(tmpfile, path)
    return path


def get_download_dir(download_dir: Optional[str] = None) -> Path:
    """Get a directory where datasets should be downloaded to.

    The download directory is determined by the following steps:

    #. If the ``download_dir`` parameter is provided, use that.
    #. If the ``CAPYMOA_DATASETS_DIR`` environment variable is set, use that.
    #. Otherwise, use the default download directory: ``./data``.

    :param download_dir: Override the download directory.
    :return: The download directory.
    """
    if download_dir is not None:
        download_dir = Path(download_dir)
    else:
        download_dir = Path(capymoa_datasets_dir())
    download_dir: Path
    download_dir.mkdir(exist_ok=True)
    return download_dir


def download_numpy_dataset(
    dataset_name: str,
    url: str,
    auto_download: bool = True,
    downloads: Path | str = capymoa_datasets_dir(),
) -> Tuple[
    Tuple[np.ndarray, np.ndarray],
    Tuple[np.ndarray, np.ndarray],
]:
    """Download, extract, and load a numpy dataset.

    Assumes the dataset has been archived in a tar.gz file with the following
    structure:

    ..  code-block:: text

        ${dataset_name}/
            train_x.npy train_y.npy test_x.npy test_y.npy

    :param dataset_name: Dataset name, used to create the directory and archive
        filename.
    :param url: URL pointing to the dataset archive.
    :param auto_download: If True, the dataset will be downloaded if it does not
        exist.
    :param output_directory: Directory to download the dataset to. Defaults to
        the CapyMOA datasets directory.
    :raises FileNotFoundError: If the dataset is not found and `auto_download`
        is False.
    :return: A tuple containing the training and testing data as numpy arrays.
    """
    path = Path(downloads) / dataset_name

    # Check if the dataset is already downloaded
    if not path.exists():
        if not auto_download:
            raise FileNotFoundError(
                f"Dataset {dataset_name} not found in {downloads}. "
                "Set auto_download=True to download it."
            )
        assert download_unpacked(url, downloads) == path

    return (
        (
            np.load(path / "train_x.npy"),
            np.load(path / "train_y.npy"),
        ),
        (
            np.load(path / "test_x.npy"),
            np.load(path / "test_y.npy"),
        ),
    )


class TensorDatasetWithTransform(
    torch.utils.data.Dataset[Tuple[torch.Tensor, torch.Tensor]]
):
    """A PyTorch dataset that applies a transformation to the data."""

    def __init__(
        self,
        data: torch.Tensor,
        targets: torch.Tensor,
        transform: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
    ):
        self.data = data
        self.targets = targets
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.data[idx]
        y = self.targets[idx]

        if self.transform:
            x = self.transform(x)

        return x, y
