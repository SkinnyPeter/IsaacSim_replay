from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass(frozen=True)
class FrameWindow:
    start: int = 0
    end: int | None = None

    @property
    def length(self) -> int | None:
        if self.end is None:
            return None
        return max(0, self.end - self.start + 1)


def load_frame_window(metadata_path: str | Path) -> FrameWindow:
    """Load optional original-frame bounds from a demo metadata YAML file."""
    path = Path(metadata_path)
    if not path.exists():
        return FrameWindow()

    with open(path, encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}

    start = int(raw.get("frame_start", 0) or 0)
    end_raw = raw.get("frame_end")
    end = None if end_raw is None else int(end_raw)

    if start < 0:
        raise ValueError(f"{path} frame_start must be >= 0")
    if end is not None and end < start:
        raise ValueError(f"{path} frame_end must be >= frame_start")

    return FrameWindow(start=start, end=end)
