"""Trusted process-local state shared by separately loaded YAML callables."""

from pathlib import Path
from typing import Any

annotation_root: Path | None = None
video_root: Path | None = None
media_index: dict[str, str] = {}
targets: dict[str, Any] = {}
