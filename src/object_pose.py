from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class ObjectPoseSequence:
    poses: np.ndarray
    is_static: bool
    source_shape: tuple[int, ...]

    def __len__(self) -> int:
        return len(self.poses)

    def __getitem__(self, frame):
        return self.poses[frame]


def load_object_pose_sequence(path: str | Path, replay_frames: int) -> ObjectPoseSequence:
    """Load an object pose file as either a static pose or per-frame trajectory."""
    raw = np.load(path)
    source_shape = tuple(raw.shape)

    if raw.shape == (4, 4):
        poses = np.broadcast_to(raw, (int(replay_frames), 4, 4))
        return ObjectPoseSequence(poses=poses, is_static=True, source_shape=source_shape)

    if raw.ndim == 3 and raw.shape[1:] == (4, 4):
        if raw.shape[0] < 1:
            raise ValueError(f"{path} must contain at least one 4x4 pose")
        if raw.shape[0] == 1:
            poses = np.broadcast_to(raw[0], (int(replay_frames), 4, 4))
            return ObjectPoseSequence(poses=poses, is_static=True, source_shape=source_shape)
        return ObjectPoseSequence(poses=raw, is_static=False, source_shape=source_shape)

    raise ValueError(
        f"{path} must have shape (4, 4) for a static pose or (N, 4, 4) for a trajectory; "
        f"got {source_shape}"
    )
