"""List released doc versions from GitHub Releases, newest first.

Every release with a docs-*.tar.gz asset attached has documentation to publish.
GitHub Releases is the source of truth for this list: there is no separate versions
file to keep in sync.
"""

import argparse
import json
import re
import subprocess
from pathlib import Path

from _common import write_tags

DOCS_ASSET_PATTERN = re.compile(r"^docs-.*\.tar\.gz$")


def filter_doc_releases(releases: list[dict]) -> list[str]:
    """Return tag names of non-draft releases with a docs asset, in input order."""
    tags = []
    for release in releases:
        if release.get("isDraft"):
            continue
        asset_names = [asset["name"] for asset in release.get("assets", [])]
        if any(DOCS_ASSET_PATTERN.match(name) for name in asset_names):
            tags.append(release["tagName"])
    return tags


def fetch_releases(repo: str) -> list[dict]:
    """Fetch releases from GitHub, newest first (the gh CLI's default order)."""
    result = subprocess.run(
        [
            "gh",
            "release",
            "list",
            "--repo",
            repo,
            "--limit",
            "1000",
            "--json",
            "tagName,isDraft,assets",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return json.loads(result.stdout)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo",
        required=True,
        help="GitHub repository to list releases from, e.g. owner/repo.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Path to write the newline-separated tag list to.",
    )
    args = parser.parse_args()

    tags = filter_doc_releases(fetch_releases(args.repo))
    write_tags(args.output, tags)


if __name__ == "__main__":
    main()
