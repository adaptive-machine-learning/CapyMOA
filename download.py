from pathlib import Path
from urllib.parse import urlparse
from urllib.request import urlretrieve
from os.path import basename
from shutil import unpack_archive, get_unpack_formats, copyfileobj, move
import gzip
from typing import Tuple

_GZIP_SUFFIX = [".gz", ".gzip"]


def unpacked_format(filename: str) -> Tuple[str | None, str]:
    for format, extensions, _ in get_unpack_formats():
        for ext in extensions:
            if filename.endswith(ext):
                return format, filename.removesuffix(ext)
    for ext in _GZIP_SUFFIX:
        if filename.endswith(ext):
            return "gzip", filename.removesuffix(ext)
    return None, filename


def unpacked_filename(filename: str) -> str:
    return unpacked_filename(filename)[1]


def is_already_downloaded(url: str, downloads: Path | str) -> bool:
    downloads = Path(downloads)
    path = downloads / unpacked_filename(url_to_filename(url))
    return path.exists()


def url_to_filename(url: str) -> str:
    return basename(urlparse(url).path)


def download_unpacked(url: str, downloads: Path | str):
    downloads = Path(downloads)
    if not downloads.exists():
        raise FileNotFoundError()
    if not downloads.is_dir():
        raise NotADirectoryError()

    tmpfile, _ = urlretrieve(url)

    # Unpack, decompress, or simply move to the path
    format, filename = unpacked_format(url_to_filename(url))
    if format == "gzip":
        with gzip.open(tmpfile, "rb") as fin:
            with open(downloads / filename, "xb") as fdst:
                copyfileobj(fin, fdst)
    elif format is not None:
        unpack_archive(tmpfile, downloads / filename, format=format)
    elif format is None:
        move(tmpfile, downloads / filename)
