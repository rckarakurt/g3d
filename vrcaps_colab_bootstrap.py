"""Minimal Colab bootstrap — no third-party deps, safe before pip install."""

from __future__ import annotations

import os
import sys
from pathlib import Path


def bootstrap() -> tuple[Path, Path]:
    repo = Path(os.environ.get("VRCAPS_REPO", "/content/g3d"))
    orbit = repo / "colab" / "orbit_coating"
    gaze = repo / "captures"
    legacy = Path("/content/vrcaps_scripts")
    if not (orbit / "coating_utils.py").exists() and (legacy / "coating_utils.py").exists():
        orbit = legacy
        gaze = legacy / "gaze"
    for path in map(str, (orbit, gaze)):
        if path not in sys.path:
            sys.path.insert(0, path)
    return orbit, gaze
