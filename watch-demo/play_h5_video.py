"""Interactive H5 camera player.

The player reads frames lazily from one or more image datasets in an H5 file.
It supports pause/resume, single-frame stepping, speed changes, and seeking
with an OpenCV trackbar.

Run from the project root:

    python watch-demo/play_h5_video.py data/h5/20250804_105719.h5

Controls:
    SPACE      pause/resume
    + or =     double playback speed
    - or _     halve playback speed
    . or right step one frame forward and pause
    , or left  step one frame backward and pause
    q or ESC   quit
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2
import h5py
import numpy as np


BASE_DIR = Path(__file__).resolve().parents[1]
DEFAULT_H5_DIR = BASE_DIR / "data" / "h5"

KNOWN_CAMERA_KEYS = {
    "aria": "observations/images/aria_rgb_cam/color",
    "oakd": "observations/images/oakd_front_view/color",
}

WINDOW_NAME = "H5 demo player"
TRACKBAR_NAME = "frame"


def latest_h5(directory: Path = DEFAULT_H5_DIR) -> Path:
    """Return the newest-looking H5 by sorted filename."""

    files = sorted(directory.glob("*.h5"))
    if not files:
        raise FileNotFoundError(f"No H5 files found in {directory}")
    return files[-1]


def discover_image_datasets(h5_file: h5py.File) -> list[str]:
    """Find likely RGB/RGBA video datasets in an H5 file."""

    datasets: list[str] = []

    def visit(name: str, obj) -> None:
        if not isinstance(obj, h5py.Dataset):
            return
        if obj.ndim >= 4 and obj.shape[-1] in (3, 4):
            datasets.append(name)

    h5_file.visititems(visit)
    return datasets


def resolve_camera_keys(h5_file: h5py.File, requested: list[str] | None) -> list[str]:
    """Map camera aliases or full dataset paths to H5 dataset keys."""

    available = discover_image_datasets(h5_file)
    if not requested:
        known_available = [key for key in KNOWN_CAMERA_KEYS.values() if key in h5_file]
        return known_available or available

    keys: list[str] = []
    for item in requested:
        key = KNOWN_CAMERA_KEYS.get(item, item)
        if key not in h5_file:
            raise KeyError(f"Camera dataset not found: {key}")
        keys.append(key)
    return keys


def read_frame(dataset: h5py.Dataset, index: int) -> np.ndarray:
    """Read one RGB/RGBA/grayscale frame and convert it to BGR for OpenCV."""

    frame = np.asarray(dataset[index])
    if frame.ndim == 2:
        return cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    if frame.shape[-1] == 4:
        return cv2.cvtColor(frame, cv2.COLOR_RGBA2BGR)
    return cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)


def stack_frames(frames: list[np.ndarray], max_panel_height: int) -> np.ndarray:
    """Resize panels to a shared height and stack them horizontally."""

    target_h = min(max_panel_height, min(frame.shape[0] for frame in frames))
    resized = []
    for frame in frames:
        if frame.shape[0] != target_h:
            scale = target_h / frame.shape[0]
            target_w = max(1, int(frame.shape[1] * scale))
            frame = cv2.resize(frame, (target_w, target_h), interpolation=cv2.INTER_AREA)
        resized.append(frame)
    return np.hstack(resized)


def draw_overlay(
    image: np.ndarray,
    frame_index: int,
    start_frame: int,
    end_frame: int,
    speed: float,
    paused: bool,
    camera_labels: list[str],
) -> np.ndarray:
    """Draw status text and a progress bar onto the displayed frame."""

    out = image.copy()
    h, w = out.shape[:2]
    cv2.rectangle(out, (0, 0), (w, 34), (0, 0, 0), thickness=-1)
    cv2.rectangle(out, (0, h - 24), (w, h), (0, 0, 0), thickness=-1)

    state = "paused" if paused else "playing"
    label = (
        f"frame {frame_index}/{end_frame} | {state} | speed x{speed:g} | "
        f"cams: {', '.join(camera_labels)}"
    )
    cv2.putText(out, label, (10, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (235, 235, 235), 1, cv2.LINE_AA)

    controls = "SPACE pause | +/- speed | ,/. step | trackbar seek | q quit"
    cv2.putText(out, controls, (10, h - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (210, 210, 210), 1, cv2.LINE_AA)

    usable = max(1, end_frame - start_frame)
    progress = (frame_index - start_frame) / usable
    bar_w = int(w * np.clip(progress, 0.0, 1.0))
    cv2.rectangle(out, (0, h - 4), (w, h), (55, 55, 55), thickness=-1)
    cv2.rectangle(out, (0, h - 4), (bar_w, h), (80, 210, 120), thickness=-1)
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Play H5 camera streams with interactive controls.")
    parser.add_argument("h5_file", nargs="?", help="Path to an H5 file. Defaults to latest file in data/h5.")
    parser.add_argument("--camera", "-c", action="append", help="Camera alias (aria/oakd) or full H5 dataset path. Repeat to show multiple.")
    parser.add_argument("--fps", type=float, default=30.0, help="Base playback FPS before speed multiplier.")
    parser.add_argument("--start", type=int, default=0, help="First frame to play.")
    parser.add_argument("--end", type=int, default=None, help="Last frame to play, inclusive.")
    parser.add_argument("--max-panel-height", type=int, default=540, help="Maximum height for each camera panel.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    h5_path = Path(args.h5_file).resolve() if args.h5_file else latest_h5()
    if not h5_path.exists():
        raise FileNotFoundError(f"H5 file not found: {h5_path}")

    speed = 1.0
    paused = False
    seek_request: int | None = None
    ignore_trackbar = False

    def on_seek(value: int) -> None:
        nonlocal seek_request
        if not ignore_trackbar:
            seek_request = value

    with h5py.File(h5_path, "r") as f:
        camera_keys = resolve_camera_keys(f, args.camera)
        if not camera_keys:
            raise RuntimeError("No image datasets found in the H5 file.")

        streams = {key: f[key] for key in camera_keys}
        n_frames = min(stream.shape[0] for stream in streams.values())
        start_frame = max(0, int(args.start))
        end_frame = n_frames - 1 if args.end is None else min(int(args.end), n_frames - 1)
        if start_frame > end_frame:
            raise ValueError(f"Empty frame range: start={start_frame}, end={end_frame}")

        camera_labels = [next((name for name, key in KNOWN_CAMERA_KEYS.items() if key == camera), camera) for camera in camera_keys]
        print(f"Opening: {h5_path}")
        print(f"Cameras: {camera_labels}")
        print(f"Frames : {start_frame}..{end_frame} of {n_frames}")
        print("Controls: SPACE pause | +/- speed | ,/. step | trackbar seek | q quit")

        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
        cv2.createTrackbar(TRACKBAR_NAME, WINDOW_NAME, start_frame, end_frame, on_seek)

        frame_index = start_frame
        last_tick = time.perf_counter()
        while True:
            if seek_request is not None:
                frame_index = int(np.clip(seek_request, start_frame, end_frame))
                seek_request = None
                paused = True

            frames = [read_frame(streams[key], frame_index) for key in camera_keys]
            image = stack_frames(frames, args.max_panel_height)
            image = draw_overlay(image, frame_index, start_frame, end_frame, speed, paused, camera_labels)

            ignore_trackbar = True
            cv2.setTrackbarPos(TRACKBAR_NAME, WINDOW_NAME, frame_index)
            ignore_trackbar = False
            cv2.imshow(WINDOW_NAME, image)

            delay_ms = max(1, int(1000.0 / max(1e-6, args.fps * speed)))
            key = cv2.waitKeyEx(20 if paused else delay_ms)

            if key in (27, ord("q")):
                break
            if key == ord(" "):
                paused = not paused
            elif key in (ord("+"), ord("=")):
                speed = min(speed * 2.0, 16.0)
            elif key in (ord("-"), ord("_")):
                speed = max(speed * 0.5, 0.0625)
            elif key in (ord("."), ord("d"), 83, 65363, 2555904):
                frame_index = min(frame_index + 1, end_frame)
                paused = True
            elif key in (ord(","), ord("a"), 81, 65361, 2424832):
                frame_index = max(frame_index - 1, start_frame)
                paused = True

            if not paused:
                now = time.perf_counter()
                if now - last_tick >= delay_ms / 1000.0:
                    frame_index += 1
                    last_tick = now
                if frame_index > end_frame:
                    frame_index = start_frame
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
