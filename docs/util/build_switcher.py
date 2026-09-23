"""Generate switcher.json for the pydata-sphinx-theme version switcher.

https://pydata-sphinx-theme.readthedocs.io/en/stable/user_guide/version-dropdown.html

Takes the list of released doc versions, newest first, and the site's base URL, and
writes the JSON manifest the theme's version-switcher dropdown fetches at page-load
time. Called from the ``website`` job in ``.github/workflows/release.yml``, which
enumerates GitHub releases that have a docs asset attached (newest first, matching
GitHub's default release ordering) and passes them here -- this script does no
sorting or GitHub API calls of its own.
"""

import argparse
import json
from pathlib import Path


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
        "--tag",
        dest="tags",
        action="append",
        required=True,
        help="A released tag with docs (e.g. v0.14.0). Pass newest first; repeat for "
        "each version. The first --tag is marked 'preferred' (stable).",
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

    switcher = build_switcher(args.tags, args.base_url)
    args.output.write_text(json.dumps(switcher, indent=2) + "\n")


if __name__ == "__main__":
    main()
