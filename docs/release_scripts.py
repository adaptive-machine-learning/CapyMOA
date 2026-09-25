"""CLI scripts used by the Release GitHub Actions workflow
(.github/workflows/release.yml) to build and publish CapyMOA's versioned
documentation site.

Subcommands:

* ``list-versions`` lists every GitHub Release that has a docs asset attached, and
  applies the retention policy (keep the last N releases of the current major
  version, and only the latest release of every earlier major version).
* ``assemble`` downloads each listed release's docs asset and extracts it into its
  own /vX.Y.Z/ folder, copying the newest into the site root.
* ``build-switcher`` writes switcher.json for the pydata-sphinx-theme version
  switcher (https://pydata-sphinx-theme.readthedocs.io/en/stable/user_guide/version-dropdown.html).
"""

import argparse
import json
import re
import shutil
import subprocess
import sys
import tarfile
import tomllib
from pathlib import Path

_PYPROJECT = Path(__file__).resolve().parents[1] / "pyproject.toml"
_TAG_RE = re.compile(r"^v(\d+)\.(\d+)\.(\d+)$")
DOCS_ASSET_PATTERN = re.compile(r"^docs-.*\.tar\.gz$")


# -- Shared helpers -----------------------------------------------------------


def read_tags(path: Path) -> list[str]:
    """Read a newline-separated list of tags, skipping blank lines."""
    return [line.strip() for line in path.read_text().splitlines() if line.strip()]


def write_tags(path: Path, tags: list[str]) -> None:
    """Write tags one per line."""
    path.write_text("".join(f"{tag}\n" for tag in tags))


def run_gh(*args: str, capture_output: bool = False) -> subprocess.CompletedProcess:
    """Run `gh <args...>`, raising if it fails.

    Captures stderr even when capture_output is False, so a failure prints
    gh's actual error message instead of just a bare CalledProcessError.
    """
    try:
        return subprocess.run(
            ["gh", *args], check=True, capture_output=capture_output, text=True
        )
    except subprocess.CalledProcessError as e:
        if e.stderr:
            print(e.stderr, file=sys.stderr)
        raise


def gh_json(*args: str) -> object:
    """Run `gh <args...>` and parse its stdout as JSON."""
    return json.loads(run_gh(*args, capture_output=True).stdout)


def parse_tag(tag: str) -> tuple[int, int, int]:
    """Parse a vMAJOR.MINOR.PATCH release tag into a comparable tuple."""
    match = _TAG_RE.match(tag)
    if not match:
        raise ValueError(f"tag {tag!r} does not match vMAJOR.MINOR.PATCH")
    major, minor, patch = match.groups()
    return (int(major), int(minor), int(patch))


def select_versions_to_publish(tags: list[str], current_line_limit: int) -> list[str]:
    """Select which releases to keep live, newest first.

    Sorts by parsed semver rather than input order, then groups by major version.
    Keeps the last `current_line_limit` releases of the current (highest) major line,
    and only the latest release of every earlier major line.
    """
    parsed = sorted(((parse_tag(tag), tag) for tag in tags), reverse=True)
    if not parsed:
        return []

    current_major = parsed[0][0][0]
    selected: list[str] = []
    seen_majors: set[int] = set()
    current_line_count = 0
    for version, tag in parsed:
        major = version[0]
        if major == current_major:
            if current_line_count < current_line_limit:
                selected.append(tag)
                current_line_count += 1
        elif major not in seen_majors:
            selected.append(tag)
            seen_majors.add(major)
    return selected


def site_base_url() -> str:
    """The canonical docs site URL, read from pyproject.toml's [project.urls]."""
    with _PYPROJECT.open("rb") as f:
        data = tomllib.load(f)
    return data["project"]["urls"]["Documentation"].rstrip("/")


# -- list-versions --------------------------------------------------------------


def filter_doc_releases(releases: list[dict]) -> list[str]:
    """Return tag names of eligible releases: not a draft, not a prerelease, and
    with a docs asset attached. Order is not meaningful; callers must sort."""
    tags = []
    for release in releases:
        if release.get("isDraft") or release.get("isPrerelease"):
            continue
        asset_names = [asset["name"] for asset in release.get("assets", [])]
        if any(DOCS_ASSET_PATTERN.match(name) for name in asset_names):
            tags.append(release["tagName"])
    return tags


def fetch_releases(repo: str) -> list[dict]:
    """Fetch all releases from GitHub, in whatever order the gh CLI returns them.

    Uses `gh api` rather than `gh release list --json assets`: the latter's
    --json flag doesn't support an `assets` field (only `gh release view`
    does), so it can't tell which releases have a docs asset attached.
    """
    raw = gh_json("api", f"repos/{repo}/releases", "--paginate")
    return [
        {
            "tagName": release["tag_name"],
            "isDraft": release["draft"],
            "isPrerelease": release["prerelease"],
            "assets": [{"name": asset["name"]} for asset in release["assets"]],
        }
        for release in raw
    ]


def _cmd_list_versions(args: argparse.Namespace) -> None:
    eligible = filter_doc_releases(fetch_releases(args.repo))
    tags = select_versions_to_publish(eligible, args.keep_current_line)
    write_tags(args.output, tags)


# -- assemble -------------------------------------------------------------------


def download_docs_asset(tag: str, repo: str, dest_dir: Path) -> Path:
    """Download tag's docs-*.tar.gz release asset into dest_dir, return its path."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    run_gh(
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
    )
    matches = sorted(dest_dir.glob("docs-*.tar.gz"))
    if len(matches) != 1:
        raise RuntimeError(
            f"expected exactly one docs-*.tar.gz asset for {tag} in {dest_dir}, "
            f"found {len(matches)}: {matches}"
        )
    return matches[0]


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


def _cmd_assemble(args: argparse.Namespace) -> None:
    assemble(read_tags(args.tags_file), args.repo, args.publish_dir, args.tmp_dir)


# -- build-switcher ---------------------------------------------------------------


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


def _cmd_build_switcher(args: argparse.Namespace) -> None:
    base_url = args.base_url or site_base_url()
    switcher = build_switcher(read_tags(args.tags_file), base_url)
    args.output.write_text(json.dumps(switcher, indent=2) + "\n")


# -- CLI --------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    p_list = subparsers.add_parser(
        "list-versions",
        help="List released doc versions from GitHub Releases, newest first.",
    )
    p_list.add_argument(
        "--repo",
        required=True,
        help="GitHub repository to list releases from, e.g. owner/repo.",
    )
    p_list.add_argument(
        "--keep-current-line",
        type=int,
        default=5,
        help="Number of releases to keep live for the current major version. "
        "Every earlier major version keeps only its latest release. Default: 5.",
    )
    p_list.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Path to write the newline-separated tag list to.",
    )
    p_list.set_defaults(func=_cmd_list_versions)

    p_assemble = subparsers.add_parser(
        "assemble", help="Download and assemble the versioned docs site tree."
    )
    p_assemble.add_argument(
        "--repo",
        required=True,
        help="GitHub repository to download release assets from, e.g. owner/repo.",
    )
    p_assemble.add_argument(
        "--tags-file",
        type=Path,
        required=True,
        help="Path to a newline-separated list of released tags, newest first "
        "(see the list-versions subcommand).",
    )
    p_assemble.add_argument(
        "--publish-dir",
        type=Path,
        default=Path("publish"),
        help="Directory to assemble the versioned site tree in.",
    )
    p_assemble.add_argument(
        "--tmp-dir",
        type=Path,
        default=Path("tmp-downloads"),
        help="Scratch directory for downloaded archives before extraction.",
    )
    p_assemble.set_defaults(func=_cmd_assemble)

    p_switcher = subparsers.add_parser(
        "build-switcher",
        help="Generate switcher.json for the pydata-sphinx-theme version switcher.",
    )
    p_switcher.add_argument(
        "--tags-file",
        type=Path,
        required=True,
        help="Path to a newline-separated list of released tags, newest first "
        "(see the list-versions subcommand). The first tag is marked 'preferred' "
        "(stable).",
    )
    p_switcher.add_argument(
        "--base-url",
        default=None,
        help="Site base URL docs are served from. Defaults to the Documentation URL "
        "in pyproject.toml's [project.urls].",
    )
    p_switcher.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Path to write switcher.json to.",
    )
    p_switcher.set_defaults(func=_cmd_build_switcher)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
