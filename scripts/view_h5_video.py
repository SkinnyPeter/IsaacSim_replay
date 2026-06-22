"""Visualize or save camera streams from an H5 file.

No Isaac Sim or OpenCV required — uses matplotlib for display, imageio for saving.

Usage:
    python3 scripts/view_h5_video.py                                    # view latest file
    python3 scripts/view_h5_video.py path/to/file.h5                    # view specific file
    python3 scripts/view_h5_video.py path/to/file.h5 --fps 15           # custom playback speed
    python3 scripts/view_h5_video.py path/to/file.h5 --save             # save to data/mp4/<id>_rgb.mp4
    python3 scripts/view_h5_video.py path/to/file.h5 --save output.mp4  # save to custom path
    python3 scripts/view_h5_video.py path/to/file.h5 --save --cam aria  # one camera only

Controls (viewer only):
    SPACE  — pause / resume
    →      — step one frame forward (while paused)
    ←      — step one frame backward (while paused)
    ESC/Q  — quit
"""

import sys
import argparse
from pathlib import Path

import h5py
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation


BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
MP4_DIR  = DATA_DIR / "mp4"

CAMERA_KEYS = [
    "observations/images/aria_rgb_cam/color",
    "observations/images/oakd_front_view/color",
]

CAMERA_LABELS = {
    "observations/images/aria_rgb_cam/color":    "Aria RGB",
    "observations/images/oakd_front_view/color": "OAK-D Front",
}


def find_latest_h5(directory: Path) -> Path:
    files = sorted(directory.glob("*.h5"))
    if not files:
        raise FileNotFoundError(f"No .h5 files found in {directory}")
    return files[-1]


def main():
    parser = argparse.ArgumentParser(description="View H5 camera streams")
    parser.add_argument("h5_file", nargs="?", help="Path to .h5 file (default: latest in ~/Desktop/data)")
    parser.add_argument("--fps",  type=float, default=30.0, help="Playback/save fps (default: 30)")
    parser.add_argument("--save", nargs="?", const="", default=None,
                        help="Save to video instead of displaying. Omit a value to auto-save to data/mp4/<id>_rgb.mp4")
    parser.add_argument("--cam",  type=str,   default=None, choices=["aria", "oakd"], help="Save only one camera: aria or oakd")
    args = parser.parse_args()

    h5_path = Path(args.h5_file) if args.h5_file else find_latest_h5(DATA_DIR / "h5")

    if not h5_path.exists():
        print(f"ERROR: file not found: {h5_path}")
        sys.exit(1)

    if args.save == "":
        MP4_DIR.mkdir(parents=True, exist_ok=True)
        args.save = str(MP4_DIR / f"{h5_path.stem}_rgb.mp4")

    print(f"Opening: {h5_path}")

    with h5py.File(h5_path, "r") as f:
        available = [k for k in CAMERA_KEYS if k in f]
        if not available:
            print("ERROR: no known camera datasets found.")
            sys.exit(1)
        # Filter to one camera if --cam is set
        if args.cam == "aria":
            available = [k for k in available if "aria" in k]
        elif args.cam == "oakd":
            available = [k for k in available if "oakd" in k]
        streams = {k: np.array(f[k]) for k in available}

    n_frames = min(s.shape[0] for s in streams.values())
    print(f"Cameras  : {[CAMERA_LABELS.get(k, k) for k in available]}")
    print(f"Frames   : {n_frames}")
    print(f"FPS      : {args.fps:.0f}")

    # ── SAVE MODE ─────────────────────────────────────────────────────────────
    if args.save:
        try:
            import imageio
        except ImportError:
            print("ERROR: imageio not installed. Run: pip install imageio imageio-ffmpeg")
            sys.exit(1)

        out_path = Path(args.save)
        print(f"Saving to: {out_path}  ({n_frames} frames @ {args.fps:.0f} fps)")

        writer = imageio.get_writer(str(out_path), fps=args.fps)
        for i in range(n_frames):
            if i % 100 == 0:
                print(f"  frame {i}/{n_frames}")
            panels = [streams[k][i] for k in available]
            # Stack side by side — resize to same height if needed
            if len(panels) > 1:
                target_h = min(p.shape[0] for p in panels)
                resized = []
                for p in panels:
                    if p.shape[0] != target_h:
                        from PIL import Image
                        pil = Image.fromarray(p)
                        w = int(p.shape[1] * target_h / p.shape[0])
                        p = np.array(pil.resize((w, target_h), Image.BILINEAR))
                    resized.append(p)
                frame = np.hstack(resized)
            else:
                frame = panels[0]
            writer.append_data(frame)
        writer.close()
        print(f"Saved: {out_path}")
        return

    print("Controls : SPACE=pause  ←/→=step  ESC/Q=quit")

    n_cams = len(available)
    fig, axes = plt.subplots(1, n_cams, figsize=(7 * n_cams, 5))
    if n_cams == 1:
        axes = [axes]
    fig.patch.set_facecolor("black")

    im_handles = []
    title_handles = []
    for ax, key in zip(axes, available):
        ax.axis("off")
        im = ax.imshow(streams[key][0])
        title = ax.set_title(CAMERA_LABELS.get(key, key), color="white", fontsize=12)
        im_handles.append(im)
        title_handles.append(title)

    state = {"frame": 0, "paused": False}
    status_text = fig.text(0.5, 0.02, "", ha="center", color="lightgray", fontsize=10)

    def update_display():
        i = state["frame"]
        for im, key in zip(im_handles, available):
            im.set_data(streams[key][i])
        label = "[PAUSED]" if state["paused"] else "playing"
        status_text.set_text(f"frame {i + 1}/{n_frames}  {label}")
        fig.canvas.draw_idle()

    def on_key(event):
        if event.key in ("escape", "q"):
            plt.close(fig)
        elif event.key == " ":
            state["paused"] = not state["paused"]
            update_display()
        elif event.key == "right":
            state["frame"] = min(state["frame"] + 1, n_frames - 1)
            update_display()
        elif event.key == "left":
            state["frame"] = max(state["frame"] - 1, 0)
            update_display()

    fig.canvas.mpl_connect("key_press_event", on_key)

    interval_ms = max(1, int(1000 / args.fps))

    def animate(_):
        if not state["paused"]:
            state["frame"] = (state["frame"] + 1) % n_frames
            update_display()

    ani = animation.FuncAnimation(fig, animate, interval=interval_ms, cache_frame_data=False)

    update_display()
    plt.tight_layout()
    plt.show()
    print("Done.")


if __name__ == "__main__":
    main()
