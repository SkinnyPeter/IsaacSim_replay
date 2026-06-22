import csv
from enum import IntEnum
from pathlib import Path

import numpy as np


class ObjectState(IntEnum):
    STATIC  = 0
    MOVING  = 1
    GRASPED = 2


_STATE_BY_NAME = {state.name.lower(): state for state in ObjectState}
_STATE_ALIASES = {
    "occluded": ObjectState.STATIC,
    "occluded_static": ObjectState.STATIC,
    "occluded_moving": ObjectState.MOVING,
}


def _parse_state(value) -> ObjectState:
    if isinstance(value, bytes):
        value = value.decode("utf-8")
    if isinstance(value, str):
        text = value.strip().lower()
        if text in _STATE_BY_NAME:
            return _STATE_BY_NAME[text]
        if text in _STATE_ALIASES:
            return _STATE_ALIASES[text]
        return ObjectState(int(float(text)))
    return ObjectState(int(value))


def _dense_states(frame_indices: list[int], states: list[ObjectState]) -> np.ndarray:
    if not frame_indices:
        return np.empty(0, dtype=np.uint8)

    dense = np.full(max(frame_indices) + 1, ObjectState.STATIC, dtype=np.uint8)
    for frame, state in zip(frame_indices, states):
        dense[frame] = np.uint8(state)
    return dense


def _load_csv_object_state(path: Path) -> np.ndarray:
    frame_indices: list[int] = []
    states: list[ObjectState] = []

    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError(f"{path} is empty")

        fields = {name.strip().lower(): name for name in reader.fieldnames}
        state_field = fields.get("state")
        frame_field = fields.get("frame") or fields.get("frame_idx") or fields.get("stem")
        if state_field is None or frame_field is None:
            raise ValueError(
                f"{path} must contain a state column and one of frame, frame_idx, or stem"
            )

        for row in reader:
            frame_text = row[frame_field].strip()
            frame_indices.append(int(Path(frame_text).stem))
            states.append(_parse_state(row[state_field]))

    return _dense_states(frame_indices, states)


def load_object_state(path: str | Path) -> np.ndarray:
    """Load per-frame object states from detector .npy arrays or motion-state CSVs."""
    path = Path(path)
    suffix = path.suffix.lower()

    if suffix == ".csv":
        return _load_csv_object_state(path)

    raw_state = np.load(path, allow_pickle=False)
    if raw_state.ndim == 1:
        return np.asarray([_parse_state(value) for value in raw_state], dtype=np.uint8)
    if raw_state.ndim != 2 or raw_state.shape[1] < 2:
        raise ValueError(f"{path} must be a 1D state array or an Nx2 frame/state array")

    frame_indices = [int(frame) for frame in raw_state[:, 0]]
    states = [_parse_state(value) for value in raw_state[:, 1]]
    return _dense_states(frame_indices, states)
