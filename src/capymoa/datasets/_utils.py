import gzip
import shutil
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Optional, Union, Tuple
import wget
from shutil import copyfileobj
from capymoa.env import capymoa_datasets_dir
from urllib.parse import urlsplit


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


def extract(in_filename: Union[Path, str]) -> Path:
    """Extract the given file.

    :param in_filename: A filename to extract with a known archive/compression/no extension.
    :raises ValueError: If the file extension is unknown.
    :return: The extracted filename without the archive/compression suffix.
    """
    in_filename = Path(in_filename)

    suffix, out_filename = identify_compressed_file(in_filename)
    out_path = in_filename.parent / out_filename

    if suffix == ".gz":
        with gzip.open(in_filename, "rb") as f_in:
            with open(out_path, "wb") as f_out:
                copyfileobj(f_in, f_out)
    else:
        raise ValueError(f"Unknown file extension: {suffix}")
    return out_path


def identify_compressed_file(path: Union[str, Path]) -> Tuple[str, str]:
    """
    Returns the name and suffix of a compressed file.

    Useful to determine the name of a file after extraction.

    >>> identify_compressed_file("file.csv.gz")
    ('.gz', 'file.csv')
    >>> identify_compressed_file("https://example.com/file.csv.gz")
    ('.gz', 'file.csv')

    :param filename: The filename or url to extract the suffix from.
    :return: A tuple containing the suffix and the extracted filename.
    """
    # Convert to string to handle Path objects and URLs
    path = Path(path)

    suffix = path.suffixes[-1]
    if suffix == ".gz":
        return suffix, path.with_suffix("").name
    else:
        raise ValueError(f"Unknown file extension: {suffix}")


def identify_compressed_hosted_file(url: str) -> Tuple[str, str]:
    """Returns the extracted filename of a given URL and its suffix.

    >>> identify_compressed_hosted_file("https://example.com/file.csv.gz")
    ('.gz', 'file.csv')
    """
    return identify_compressed_file(urlsplit(url).path)


def is_already_downloaded(url: str, output_directory: Union[str, Path]) -> bool:
    """Check if a file has already been downloaded.

    This function checks if the url has already been downloaded by checking if the
    extracted file exists in the output directory.
    """
    return (Path(output_directory) / identify_compressed_hosted_file(url)[1]).exists()


def download_extract(url: str, output_directory: Path) -> Path:
    """Download and extract a file.

    :param url: URL pointing to the file to download.
    :param output_directory: A directory to download the file to.
    :return: The path to the extracted file.
    """
    output_directory = Path(output_directory)
    if not output_directory.exists():
        raise FileNotFoundError(f"Output directory {output_directory} does not exist.")
    if not output_directory.is_dir():
        raise NotADirectoryError(
            f"Output directory {output_directory} is not a directory."
        )

    with TemporaryDirectory() as working_directory:
        working_directory = Path(working_directory)
        archive = wget.download(url, working_directory.absolute().as_posix())
        extracted = extract(archive)
        out_filename = output_directory / extracted.name
        shutil.move(extracted, out_filename)
    return out_filename
