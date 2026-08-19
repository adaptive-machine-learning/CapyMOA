"""Generic loading of datasets directly from OpenML by their numeric dataset id."""

import json
from pathlib import Path
from typing import List, Optional, Union
from urllib.error import HTTPError, URLError
from urllib.request import urlopen

from capymoa.datasets._utils import (
    download_unpacked,
    get_download_dir,
    infer_unpacked_path,
    is_already_downloaded,
)
from capymoa.stream import Stream, stream_from_file

_OPENML_API = "https://www.openml.org/api/v1/json/data/{id}"


def _fetch_description(openml_id: int, cache_dir: Path, auto_download: bool) -> dict:
    """Return the OpenML API JSON description for a dataset id.

    Cached as ``<cache_dir>/description.json`` so repeated loads never re-hit
    the OpenML API once fetched once.
    """
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
        message = f"OpenML API returned HTTP {e.code} for dataset {openml_id}."
        try:
            body = json.loads(e.read())
            message = body["error"]["message"]
        except (json.JSONDecodeError, KeyError, ValueError):
            pass
        raise ValueError(
            f"Could not fetch OpenML dataset {openml_id}: {message}"
        ) from e
    except URLError as e:
        raise ConnectionError(f"Could not reach the OpenML API: {e.reason}") from e

    description = payload["data_set_description"]
    cache_path.write_text(json.dumps(description))
    return description


def _attribute_names(path: Path) -> List[str]:
    """Parse ``@attribute <name> <type>`` lines from an ARFF file, up to ``@data``."""
    names = []
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            stripped = line.strip()
            if stripped.lower().startswith("@data"):
                break
            if not stripped.lower().startswith("@attribute"):
                continue
            rest = stripped[len("@attribute") :].strip()
            if rest.startswith("'") or rest.startswith('"'):
                quote = rest[0]
                end = rest.find(quote, 1)
                name = rest[1:end]
            else:
                name = rest.split(None, 1)[0]
            names.append(name)
    return names


def load_openml_dataset(
    openml_id: int,
    directory: Optional[Union[str, Path]] = None,
    auto_download: bool = True,
) -> Stream:
    """Load any OpenML dataset by numeric id as a :class:`~capymoa.stream.Stream`.

    Fetches the dataset's metadata from the OpenML API (cached to disk after
    the first call, so subsequent loads never re-hit OpenML), downloads its
    ARFF file (cached under ``<download_dir>/openml/<openml_id>/``), resolves
    the target column declared by OpenML's ``default_target_attribute``, and
    returns a :class:`~capymoa.stream.Stream`.

    >>> from capymoa.datasets import load_openml_dataset
    >>> stream = load_openml_dataset(1169)  # doctest: +SKIP

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

    names = _attribute_names(path)
    try:
        class_index = names.index(target)
    except ValueError:
        raise ValueError(
            f"Target attribute {target!r} for OpenML dataset {openml_id} was not "
            f"found among the ARFF attributes in {path}."
        ) from None

    return stream_from_file(
        path,
        dataset_name=description.get("name", f"openml_{openml_id}"),
        class_index=class_index,
    )
