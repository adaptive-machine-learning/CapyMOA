"""Download every released docs package and assemble the versioned site tree.

For each tag (newest first), downloads its docs-<tag>.tar.gz GitHub Release asset and
extracts it into <publish-dir>/<tag>/. The newest tag is also copied to the top of
<publish-dir>, so the site root doubles as "latest".
"""

import argparse
import shutil
import subprocess
import tarfile
from pathlib import Path

from _common import read_tags


def download_docs_asset(tag: str, repo: str, dest_dir: Path) -> Path:
    """Download tag's docs-*.tar.gz release asset into dest_dir, return its path."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            "gh",
            "release",
            "download",
            tag,
            "--repo",
            repo,
            "--pattern",
            "docs-*.tar.gz",
            "--dir",
            str(dest_dir),
            "--clobber",
        ],
        check=True,
    )
    (archive,) = dest_dir.glob("docs-*.tar.gz")
    return archive


def extract_docs(archive: Path, dest_dir: Path) -> None:
    dest_dir.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive) as tar:
        tar.extractall(dest_dir, filter="data")


def assemble(tags: list[str], repo: str, publish_dir: Path, tmp_dir: Path) -> None:
    for i, tag in enumerate(tags):
        archive = download_docs_asset(tag, repo, tmp_dir / tag)
        version_dir = publish_dir / tag
        extract_docs(archive, version_dir)
        if i == 0:
            # Newest tag doubles as the site root ("latest").
            shutil.copytree(version_dir, publish_dir, dirs_exist_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo",
        required=True,
        help="GitHub repository to download release assets from, e.g. owner/repo.",
    )
    parser.add_argument(
        "--tags-file",
        type=Path,
        required=True,
        help="Path to a newline-separated list of released tags, newest first "
        "(see list_doc_versions.py).",
    )
    parser.add_argument(
        "--publish-dir",
        type=Path,
        default=Path("publish"),
        help="Directory to assemble the versioned site tree in.",
    )
    parser.add_argument(
        "--tmp-dir",
        type=Path,
        default=Path("tmp-downloads"),
        help="Scratch directory for downloaded archives before extraction.",
    )
    args = parser.parse_args()

    assemble(read_tags(args.tags_file), args.repo, args.publish_dir, args.tmp_dir)


if __name__ == "__main__":
    main()
