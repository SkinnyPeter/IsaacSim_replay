"""Inspect H5 structure and optionally plot joint or EEF trajectories.

Run from the project root:

    python watch-demo/inspect_h5.py data/h5/20250804_105719.h5
    python watch-demo/inspect_h5.py data/h5/20250804_105719.h5 --plot-joints

The structure view is text-only. Joint plotting uses matplotlib and opens a
normal interactive figure window.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import numpy as np


BASE_DIR = Path(__file__).resolve().parents[1]
DEFAULT_H5 = BASE_DIR / "data" / "h5" / "20250804_105719.h5"

DEFAULT_JOINT_KEYS = [
    "observations/qpos_arm",
    "observations/qpos_arm_left",
    "observations/qpos_arm_right",
    "observations/qpos_hand",
    "observations/qpos_hand_left",
    "observations/qpos_hand_right",
    "actions_arm",
    "actions_arm_left",
    "actions_arm_right",
    "actions_hand",
    "actions_hand_left",
    "actions_hand_right",
]


def dataset_summary(dataset: h5py.Dataset) -> str:
    """Return one compact line describing a dataset."""

    return f"shape={tuple(dataset.shape)} dtype={dataset.dtype}"


def print_structure(h5_path: Path, preview: bool = False) -> None:
    """Print groups, datasets, shapes, dtypes, and optional first-row previews."""

    with h5py.File(h5_path, "r") as f:
        print(f"H5 file: {h5_path}")
        print(f"Top-level keys: {list(f.keys())}")

        def visit(name: str, obj) -> None:
            if isinstance(obj, h5py.Group):
                print(f"GROUP   {name}")
                return
            if isinstance(obj, h5py.Dataset):
                print(f"DATASET {name}  {dataset_summary(obj)}")
                if preview and obj.ndim >= 1 and obj.shape[0] > 0:
                    row = np.asarray(obj[0])
                    flat = row.reshape(-1)
                    shown = np.array2string(flat[:12], precision=4, separator=", ")
                    suffix = " ..." if flat.size > 12 else ""
                    print(f"        first: {shown}{suffix}")

        f.visititems(visit)


def find_plot_keys(h5_file: h5py.File, requested: list[str] | None) -> list[str]:
    """Pick numeric trajectory datasets to plot."""

    if requested:
        missing = [key for key in requested if key not in h5_file]
        if missing:
            raise KeyError(f"Dataset(s) not found: {missing}")
        return requested
    return [key for key in DEFAULT_JOINT_KEYS if key in h5_file]


def load_2d_dataset(h5_file: h5py.File, key: str, max_frames: int | None) -> np.ndarray:
    """Load a numeric trajectory and flatten non-time dimensions."""

    data = np.asarray(h5_file[key][:max_frames])
    if data.ndim == 1:
        data = data[:, None]
    elif data.ndim > 2:
        data = data.reshape(data.shape[0], -1)
    if not np.issubdtype(data.dtype, np.number):
        raise TypeError(f"Dataset is not numeric and cannot be plotted: {key}")
    return data


def plot_trajectories(h5_path: Path, keys: list[str] | None, max_frames: int | None) -> None:
    """Plot selected joint or EEF trajectories."""

    import matplotlib.pyplot as plt

    with h5py.File(h5_path, "r") as f:
        plot_keys = find_plot_keys(f, keys)
        if not plot_keys:
            print("No default joint/action datasets found. Use --key <dataset> to choose one.")
            return

        fig, axes = plt.subplots(len(plot_keys), 1, figsize=(12, max(3, 2.8 * len(plot_keys))), sharex=True)
        if len(plot_keys) == 1:
            axes = [axes]

        for ax, key in zip(axes, plot_keys):
            data = load_2d_dataset(f, key, max_frames=max_frames)
            for dim in range(data.shape[1]):
                ax.plot(data[:, dim], linewidth=1.0, label=f"d{dim}")
            ax.set_title(f"{key}  shape={data.shape}")
            ax.set_ylabel("value")
            ax.grid(True, alpha=0.25)
            if data.shape[1] <= 12:
                ax.legend(loc="upper right", fontsize=8, ncols=min(4, data.shape[1]))

        axes[-1].set_xlabel("frame")
        fig.tight_layout()
        plt.show()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect H5 structure and plot joint trajectories.")
    parser.add_argument("h5_file", nargs="?", default=str(DEFAULT_H5), help="Path to H5 file.")
    parser.add_argument("--preview", action="store_true", help="Print first-row previews for datasets.")
    parser.add_argument("--plot-joints", action="store_true", help="Open trajectory plots for default joint/action datasets.")
    parser.add_argument("--key", action="append", help="Dataset key to plot. Repeat for multiple keys.")
    parser.add_argument("--max-frames", type=int, default=None, help="Limit plotted frames for faster inspection.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    h5_path = Path(args.h5_file).resolve()
    if not h5_path.exists():
        raise FileNotFoundError(f"H5 file not found: {h5_path}")

    print_structure(h5_path, preview=args.preview)
    if args.plot_joints or args.key:
        plot_trajectories(h5_path, keys=args.key, max_frames=args.max_frames)


if __name__ == "__main__":
    main()
