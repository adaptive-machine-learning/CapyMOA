"""Run invoke commands as build step in hatchling build system."""

import subprocess
import sys
from typing import Any

from hatchling.builders.hooks.plugin.interface import BuildHookInterface


class InvokeBuildHook(BuildHookInterface):
    def initialize(self, version: str, build_data: dict[str, Any]) -> None:
        for task in ("build.download-moa", "build.stubs"):
            subprocess.run(
                [sys.executable, "-m", "invoke", task],
                cwd=self.root,
                check=True,
            )
