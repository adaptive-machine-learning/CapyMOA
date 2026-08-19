"""Generic loading of datasets directly from OpenML by their numeric dataset id."""

import json
from pathlib import Path
from typing import Optional, Union
from urllib.error import HTTPError, URLError
from urllib.request import urlopen

from moa.streams import ArffFileStream

from capymoa.datasets._utils import (
    download_unpacked,
    get_download_dir,
    infer_unpacked_path,
    is_already_downloaded,
)
from capymoa.stream import ARFFStream, Stream

_OPENML_API = "https://www.openml.org/api/v1/json/data/{id}"


def _fetch_description(openml_id: int, cache_dir: Path, auto_download: bool) -> dict:
    # Cached as `<cache_dir>/description.json` so repeated loads never re-hit
    # the OpenML API once fetched once.
    cache_path = cache_dir / "description.json"
    if cache_path.exists():
        return json.loads(cache_path.read_text())

    if not auto_download:
        raise FileNotFoundError(
            f"OpenML metadata for dataset {openml_id} not cached in {cache_dir}. "
            "Set auto_download=True to fetch it."
        )

    url = _OPENML_API.format(id=openml_id)
    try:
        with urlopen(url) as response:
            payload = json.load(response)
    except HTTPError as e:
        raise ValueError(
            f"Could not fetch OpenML dataset {openml_id}: HTTP {e.code} {e.reason}"
        ) from e
    except URLError as e:
        raise ConnectionError(f"Could not reach the OpenML API: {e.reason}") from e

    description = payload["data_set_description"]
    cache_path.write_text(json.dumps(description))
    return description


def _resolve_class_index(path: Path, target: str) -> int:
    # Let MOA parse the ARFF header (it has to anyway) rather than hand-rolling
    # an ARFF parser, and find the 0-based index of the named target attribute.
    header = ArffFileStream(str(path), -1).getHeader()
    for i in range(header.numAttributes()):
        if str(header.attribute(i).name()) == target:
            return i
    raise ValueError(
        f"Target attribute {target!r} was not found among the ARFF attributes in {path}."
    )


def load_openml_dataset(
    openml_id: int,
    directory: Optional[Union[str, Path]] = None,
    auto_download: bool = True,
) -> Stream:
    """Load any OpenML dataset by numeric id as a :class:`~capymoa.stream.Stream`.

    >>> from capymoa.datasets import load_openml_dataset
    >>> stream = load_openml_dataset(61)  # iris
    >>> stream.next_instance()
    LabeledInstance(
        Schema(iris),
        x=[5.1 3.5 1.4 0.2],
        y_index=0,
        y_label='Iris-setosa'
    )

    :param openml_id: The numeric OpenML dataset id, e.g. ``1169`` for
        https://www.openml.org/d/1169.
    :param directory: Where downloads are stored. Defaults to
        :func:`capymoa.datasets.get_download_dir`.
    :param auto_download: Fetch metadata / download the dataset if missing.
    :raises ValueError: If the dataset is not stored as ARFF on OpenML, or its
        target attribute is missing, ambiguous, or not found in the ARFF file.
    :raises ConnectionError: If the OpenML API cannot be reached.
    :raises FileNotFoundError: If the metadata or dataset is missing locally
        and ``auto_download`` is False.
    """
    # Cache metadata and the ARFF file together, namespaced by id, so repeated
    # loads for the same dataset never re-download or re-hit the OpenML API,
    # and different ids never collide on a shared filename.
    dataset_dir = get_download_dir(directory) / "openml" / str(openml_id)
    dataset_dir.mkdir(parents=True, exist_ok=True)

    description = _fetch_description(openml_id, dataset_dir, auto_download)

    if description.get("format", "").upper() != "ARFF":
        raise ValueError(
            f"OpenML dataset {openml_id} is not available in ARFF format "
            f"(got {description.get('format')!r})."
        )

    target = description.get("default_target_attribute")
    if not target or "," in target:
        raise ValueError(
            f"OpenML dataset {openml_id} does not have a single target attribute "
            f"(default_target_attribute={target!r})."
        )

    url = description["url"]
    if not is_already_downloaded(url, dataset_dir):
        if not auto_download:
            raise FileNotFoundError(
                f"OpenML dataset {openml_id} not found in {dataset_dir}. "
                "Set auto_download=True to download it."
            )
        download_unpacked(url, dataset_dir)
    path = infer_unpacked_path(url, dataset_dir)

    class_index = _resolve_class_index(path, target)
    return ARFFStream(path=path, class_index=class_index)
