"""Generate switcher.json for the pydata-sphinx-theme version switcher.

https://pydata-sphinx-theme.readthedocs.io/en/stable/user_guide/version-dropdown.html

Takes the list of released doc versions, newest first, and the site's base URL, and
writes the JSON manifest the theme's version-switcher dropdown fetches at page-load
time. The tag list comes from list_doc_versions.py; this script does no sorting or
GitHub API calls of its own.
"""

import argparse
import json
from pathlib import Path

from _common import read_tags


def build_switcher(tags: list[str], base_url: str) -> list[dict[str, object]]:
    base_url = base_url.rstrip("/")
    entries = []
    for i, tag in enumerate(tags):
        name = tag.removeprefix("v")
        entry: dict[str, object] = {
            "name": f"{name} (stable)" if i == 0 else name,
            "version": tag,
            "url": f"{base_url}/{tag}/",
        }
        if i == 0:
            entry["preferred"] = True
        entries.append(entry)
    return entries


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--tags-file",
        type=Path,
        required=True,
        help="Path to a newline-separated list of released tags, newest first "
        "(see list_doc_versions.py). The first tag is marked 'preferred' (stable).",
    )
    parser.add_argument(
        "--base-url",
        default="https://capymoa.org",
        help="Site base URL docs are served from.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Path to write switcher.json to.",
    )
    args = parser.parse_args()

    switcher = build_switcher(read_tags(args.tags_file), args.base_url)
    args.output.write_text(json.dumps(switcher, indent=2) + "\n")


if __name__ == "__main__":
    main()
