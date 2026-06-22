from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path

import h5py
import numpy as np

from src.simulator.quat_utils import detect_quaternion_order


class RobotSetup(str, Enum):
    MONO = "mono"
    DUAL = "dual"


DATA_PATH = Path(__file__).resolve().parents[2] / "data"
H5_DIR = DATA_PATH / "h5"

# IMPORTANT REMARK: For mono, qpos will be given to both left_arm and right_arm
# we assume one of the two is disabled on the simulation
REQUIRED_KEYS: dict[RobotSetup, set[str]] = {
    RobotSetup.MONO: {
        "actions_arm",
        "actions_hand",
        "observations/images/aria_rgb_cam/color",
        "observations/qpos_arm",
        "observations/qpos_hand",
    },
    RobotSetup.DUAL: {
        "actions_arm_left",
        "actions_arm_right",
        "actions_hand_left",
        "actions_hand_right",
        "observations/images/aria_rgb_cam/color",
        "observations/images/oakd_front_view/color",
        "observations/qpos_arm_left",
        "observations/qpos_arm_right",
        "observations/qpos_hand_left",
        "observations/qpos_hand_right",
    },
}

@dataclass
class ReplayData:
    right_arm: np.ndarray
    left_arm: np.ndarray
    right_hand: np.ndarray | None
    left_hand: np.ndarray | None
    n_frames: int
    structure: RobotSetup

def resolve_h5_path(sample_id_or_path: str | Path) -> Path:
    """
    Resolve a sample ID, filename, or path to an absolute .h5 file path.
    """
    candidate = Path(sample_id_or_path)

    if candidate.exists():
        return candidate

    if candidate.suffix == ".h5":
        h5_path = H5_DIR / candidate.name
    else:
        h5_path = H5_DIR / f"{candidate.name}.h5"

    if not h5_path.exists():
        raise FileNotFoundError(f"H5 file not found: {h5_path}")

    return h5_path

def _list_h5_datasets(h5_path: Path) -> set[str]:
    """
    Return all dataset paths present in an HDF5 file.
    """
    datasets = set()

    with h5py.File(h5_path, "r") as f:
        def visitor(name, obj):
            if isinstance(obj, h5py.Dataset):
                datasets.add(name)

        f.visititems(visitor)

    return datasets

def _detect_h5_structure(h5_path: Path) -> RobotSetup:
    datasets = _list_h5_datasets(h5_path)

    for setup, required in REQUIRED_KEYS.items():
        if not (required - datasets):
            return setup

    missing_per_setup = {
        setup.value: sorted(required - datasets)
        for setup, required in REQUIRED_KEYS.items()
    }
    error_lines = [f"H5 file does not match any known structure: {h5_path}\n"]
    for name, missing in missing_per_setup.items():
        error_lines.append(f"Missing for {name}:")
        error_lines.extend(f"  - {k}" for k in missing)
    error_lines.append("\nAvailable datasets:")
    error_lines.extend(f"  - {k}" for k in sorted(datasets))

    raise RuntimeError("\n".join(error_lines))

def load_replay_h5(sample_id_or_path: str | Path) -> ReplayData:
    """
    Load arm and hand trajectories from an H5 file into a ReplayData.
    """
    h5_path = resolve_h5_path(sample_id_or_path)
    structure = _detect_h5_structure(h5_path)

    with h5py.File(h5_path, "r") as f:
        if structure == RobotSetup.MONO:
            arm = np.array(f["observations/qpos_arm"])
            hand = np.array(f["observations/qpos_hand"])

            right_arm = arm
            left_arm = arm
            right_hand = hand
            left_hand = hand

        elif structure == RobotSetup.DUAL:
            right_arm = np.array(f["observations/qpos_arm_right"])
            left_arm = np.array(f["observations/qpos_arm_left"])
            right_hand = np.array(f["observations/qpos_hand_right"])
            left_hand = np.array(f["observations/qpos_hand_left"])

        else:
            raise RuntimeError(f"Unsupported H5 structure: {structure}")


    right_arm = detect_quaternion_order(right_arm, "right")
    left_arm = detect_quaternion_order(left_arm, "left")

    n_frames = min(len(right_arm), len(left_arm))

    if right_hand is not None:
        n_frames = min(n_frames, len(right_hand))
    if left_hand is not None:
        n_frames = min(n_frames, len(left_hand))

    print("\n===== H5 DATA =====")
    print("file              :", h5_path)
    print("structure         :", structure)
    print("right_arm shape   :", right_arm.shape)
    print("left_arm shape    :", left_arm.shape)
    if right_hand is not None:
        print("right_hand shape  :", right_hand.shape)
    if left_hand is not None:
        print("left_hand shape   :", left_hand.shape)
    print("n_frames          :", n_frames)

    return ReplayData(
        right_arm=right_arm,
        left_arm=left_arm,
        right_hand=right_hand,
        left_hand=left_hand,
        n_frames=n_frames,
        structure=structure,
    )