"""Shared helpers for the docs/release_scripts/ CLI scripts."""

from pathlib import Path


def read_tags(path: Path) -> list[str]:
    """Read a newline-separated list of tags, skipping blank lines."""
    return [line.strip() for line in path.read_text().splitlines() if line.strip()]


def write_tags(path: Path, tags: list[str]) -> None:
    """Write tags one per line."""
    path.write_text("".join(f"{tag}\n" for tag in tags))
