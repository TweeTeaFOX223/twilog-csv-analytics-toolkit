from __future__ import annotations

import sys
from pathlib import Path


def _bundle_root() -> Path:
    if getattr(sys, "frozen", False) and hasattr(sys, "_MEIPASS"):
        return Path(sys._MEIPASS)  # type: ignore[attr-defined]
    return Path(__file__).resolve().parent.parent


def get_templates_dir() -> Path:
    return _bundle_root() / "app" / "templates"


def get_static_dir() -> Path:
    return _bundle_root() / "app" / "static"
