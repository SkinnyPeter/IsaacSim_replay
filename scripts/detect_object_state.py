"""Detect object state from recorded EEF and object-center pixel trajectories.

For a given recording ID the script:
  1. Loads camLeft_rightEEF_pos.npy  (shape N×3, cols: frame_idx, px, py)
     and obj_center/<object_name>/centers.npy  (shape N×2, cols: px, py)
  2. Smooths both position arrays with a uniform window (config: smooth_window)
  3. Grasp detection (state 2): EEF↔center distance < pos_threshold AND
     EEF↔center speed difference < motion_threshold for min_grasp_duration frames
  4. Remaining frames: object center speed < static_threshold → 0 (static), else 1 (moving)
  5. Saves object_state.npy into outputs/<id>/object_state/  (cols: frame_idx, state)
  6. Optionally writes an annotated MP4 from the H5 aria_rgb_cam frames showing
     both pixel positions, frame number, and per-frame state label.

Usage:
    python scripts/detect_object_state.py <id>
    python scripts/detect_object_state.py <id> --config config/object_state_config.yaml
    python scripts/detect_object_state.py <id> --no-video
"""

import argparse
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from src.object_state import ObjectState

import cv2
import h5py
import numpy as np
import yaml
from scipy.ndimage import gaussian_filter1d

# ── helpers ──────────────────────────────────────────────────────────────────

def smooth(arr: np.ndarray, window: int) -> np.ndarray:
    """Gaussian filter along axis 0, applied column-wise. window is the sigma in frames."""
    if window <= 1:
        return arr.copy()
    out = np.empty_like(arr, dtype=np.float64)
    for col in range(arr.shape[1]):
        out[:, col] = gaussian_filter1d(arr[:, col].astype(np.float64), sigma=window)
    return out


def pixel_motion(pos: np.ndarray, half_window: int) -> np.ndarray:
    """Per-frame average pixel speed: mean abs frame-to-frame distance over ±half_window (px/frame)."""
    n = len(pos)
    step_dist = np.sqrt(((np.diff(pos, axis=0)) ** 2).sum(axis=1))  # shape (n-1,)
    motion = np.zeros(n)
    for t in range(n):
        lo = max(0, t - half_window)
        hi = min(n - 1, t + half_window)
        if hi > lo:
            motion[t] = step_dist[lo:hi].sum() / (hi - lo)
    return motion


# ── drawing ───────────────────────────────────────────────────────────────────

def _dot(img, px, py, color_bgr, radius=6):
    cv2.circle(img, (int(round(px)), int(round(py))), radius, color_bgr, -1, cv2.LINE_AA)


def _threshold_circle(img, px, py, radius_px, pos_ok):
    """Draw a semi-transparent circle of radius radius_px around (px, py)."""
    color = (0, 220, 0) if pos_ok else (0, 0, 220)
    overlay = img.copy()
    cv2.circle(overlay, (int(round(px)), int(round(py))), int(round(radius_px)),
               color, 2, cv2.LINE_AA)
    cv2.addWeighted(overlay, 0.45, img, 0.55, 0, img)


_STATE_COLORS = {
    ObjectState.STATIC:  (180, 180, 180),
    ObjectState.MOVING:  (50,  200, 220),
    ObjectState.GRASPED: (50,  220,  50),
}
_STATE_LABELS = {
    ObjectState.STATIC:  "STATIC",
    ObjectState.MOVING:  "MOVING",
    ObjectState.GRASPED: "GRASPED",
}


def _annotate(img, t, total, dist, speed, obj_speed, pos_ok, vel_ok, state, eef_x, eef_y, cx, cy, static_thresh):
    h = img.shape[0]

    po_color    = (0, 220, 0) if pos_ok else (0, 0, 220)
    ve_color    = (0, 220, 0) if vel_ok else (0, 0, 220)
    ob_color    = (0, 220, 0) if obj_speed < static_thresh else (0, 0, 220)
    state_color = _STATE_COLORS[state]

    cv2.putText(img, f"{t}/{total}",             (8, 18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5,  (240, 240, 240), 1, cv2.LINE_AA)
    cv2.putText(img, f"POS {dist:.0f}px",        (8, 38),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, po_color, 1, cv2.LINE_AA)
    cv2.putText(img, f"SPD DIFF {speed:.1f}px",  (8, 56),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, ve_color, 1, cv2.LINE_AA)
    cv2.putText(img, f"OBJ SPD {obj_speed:.1f}px", (8, 74),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, ob_color, 1, cv2.LINE_AA)
    cv2.putText(img, _STATE_LABELS[state],        (8, 92),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5,  state_color, 1, cv2.LINE_AA)

    lx = int(round(cx)) + 10
    ly = int(round(cy)) - 10
    cv2.putText(img, _STATE_LABELS[state][0], (lx, ly),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, state_color, 2, cv2.LINE_AA)


def state_phases(states: np.ndarray) -> list[tuple[int, int, int]]:
    """Return list of (state, start_frame, end_frame) for each contiguous run."""
    phases = []
    i = 0
    n = len(states)
    while i < n:
        s = states[i]
        j = i + 1
        while j < n and states[j] == s:
            j += 1
        phases.append((int(s), i, j - 1))
        i = j
    return phases


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Detect object state from pixel trajectories")
    parser.add_argument("id", nargs="?", help="Recording ID (e.g. 20250804_105719); overrides config")
    parser.add_argument("--config", default="config/object_state_config.yaml",
                        help="Path to object_state_config.yaml")
    parser.add_argument("--no-video", action="store_true", help="Skip video generation")
    args = parser.parse_args()

    cfg_path = BASE_DIR / args.config
    with open(cfg_path) as f:
        g = yaml.safe_load(f)

    recording_id = args.id or g.get("id")
    if not recording_id:
        sys.exit("ERROR: no recording ID — set 'id' in object_state_config.yaml or pass it as a CLI argument")

    output_dir  = BASE_DIR / g.get("output_dir", "outputs/") / recording_id
    data_dir    = BASE_DIR / g.get("data_dir", "data/h5/")
    object_name = g.get("object_name", "yellow_rubber_duck")
    smooth_win  = int(g.get("smooth_window", 15))
    pos_thresh  = float(g.get("pos_threshold", 80.0))
    motion_half    = int(g.get("motion_window", 15))
    motion_thresh  = float(g.get("motion_threshold", 5.0))
    static_thresh  = float(g.get("static_threshold", 2.0))
    static_half    = int(g.get("static_motion_window", 10))
    max_gap        = int(g.get("max_grasp_gap", 0))
    min_dur     = int(g.get("min_grasp_duration", 15))
    vid_cfg     = g.get("video", {})
    make_video  = vid_cfg.get("enabled", True) and not args.no_video
    fps         = float(vid_cfg.get("fps", 30))

    # ── load positions ────────────────────────────────────────────────────────
    eef_path     = output_dir / "EEF_pos" / "camLeft_rightEEF_pos.npy"
    centers_path = output_dir / "obj_center" / object_name / "centers.npy"

    if not eef_path.exists():
        sys.exit(f"ERROR: EEF pos not found: {eef_path}")
    if not centers_path.exists():
        sys.exit(f"ERROR: centers not found: {centers_path}")

    eef_raw     = np.load(eef_path)                         # (N, 3): frame_idx, px, py
    centers_raw = np.load(centers_path, allow_pickle=True)  # (N, 2): px, py

    if eef_raw.ndim != 2 or eef_raw.shape[1] != 3:
        sys.exit(f"Unexpected EEF shape {eef_raw.shape}, expected (N, 3)")
    if centers_raw.ndim != 2 or centers_raw.shape[1] != 2:
        sys.exit(f"Unexpected centers shape {centers_raw.shape}, expected (N, 2)")

    n_frames    = min(len(eef_raw), len(centers_raw))
    eef_px      = eef_raw[:n_frames, 1:3].astype(np.float64)   # (N, 2)
    centers_raw = centers_raw[:n_frames].astype(np.float64)     # (N, 2)

    print(f"Frames          : {n_frames}")
    print(f"Smooth window   : {smooth_win}")
    print(f"Pos threshold   : {pos_thresh:.1f} px")
    print(f"Motion half-win : {motion_half} frames (summed abs dist over ±{motion_half} frames)")
    print(f"Motion threshold: {motion_thresh:.3f} px/frame (EEF vs center avg speed diff)")
    print(f"Static threshold: {static_thresh:.3f} px/frame (object center speed)")
    print(f"Static half-win : {static_half} frames")
    print(f"Max grasp gap   : {max_gap} frames")
    print(f"Min duration    : {min_dur} frames")

    # ── smooth ────────────────────────────────────────────────────────────────
    eef_sm     = smooth(eef_px, smooth_win)
    centers_sm = smooth(centers_raw, smooth_win)

    # ── conditions ────────────────────────────────────────────────────────────
    diff   = eef_sm - centers_sm
    dist   = np.sqrt((diff ** 2).sum(axis=1))
    pos_ok = dist < pos_thresh

    motion_eef  = pixel_motion(eef_sm, motion_half)
    motion_cen  = pixel_motion(centers_sm, motion_half)
    motion_cen_static = pixel_motion(centers_sm, static_half)
    motion_diff = np.abs(motion_eef - motion_cen)  # total px over window
    motion_ok   = motion_diff < motion_thresh

    # hysteresis: raw condition must hold for min_dur consecutive frames to trigger,
    # but drops to 0 immediately when either condition fails
    raw = pos_ok & motion_ok

    # gap fill: short no-grasp runs flanked by grasp on both sides are treated as errors
    if max_gap > 0:
        i = 0
        while i < n_frames:
            if not raw[i]:
                j = i
                while j < n_frames and not raw[j]:
                    j += 1
                gap_len = j - i
                if gap_len <= max_gap and i > 0 and j < n_frames:
                    raw[i:j] = True
                i = j
            else:
                i += 1

    # ── grasp hysteresis → state 2 ────────────────────────────────────────────
    obj_state = np.zeros(n_frames, dtype=np.uint8)
    consec = 0
    run_start = 0
    for t in range(n_frames):
        if raw[t]:
            if consec == 0:
                run_start = t
            consec += 1
            if consec == min_dur:
                obj_state[run_start:t + 1] = ObjectState.GRASPED  # backfill start of run
            elif consec > min_dur:
                obj_state[t] = ObjectState.GRASPED
        else:
            consec = 0

    # ── static / moving for non-grasped frames ────────────────────────────────
    for t in range(n_frames):
        if obj_state[t] != ObjectState.GRASPED:
            obj_state[t] = ObjectState.STATIC if motion_cen_static[t] < static_thresh else ObjectState.MOVING

    # ── save ──────────────────────────────────────────────────────────────────
    state_dir = output_dir / "object_state"
    state_dir.mkdir(parents=True, exist_ok=True)
    state_path = state_dir / "object_state.npy"

    frame_indices = eef_raw[:n_frames, 0].astype(np.int64)
    state_array = np.stack([frame_indices, obj_state.astype(np.int64)], axis=1)
    np.save(state_path, state_array)

    n_static  = int((obj_state == ObjectState.STATIC).sum())
    n_moving  = int((obj_state == ObjectState.MOVING).sum())
    n_grasped = int((obj_state == ObjectState.GRASPED).sum())
    print(f"\nStatic frames  : {n_static}  / {n_frames}  ({100*n_static/n_frames:.1f}%)")
    print(f"Moving frames  : {n_moving}  / {n_frames}  ({100*n_moving/n_frames:.1f}%)")
    print(f"Grasped frames : {n_grasped} / {n_frames}  ({100*n_grasped/n_frames:.1f}%)")
    print(f"Saved state    : {state_path}")

    phases_path = state_dir / "object_state_phases.txt"
    phases = state_phases(obj_state)
    with open(phases_path, "w") as fh:
        for s, start, end in phases:
            fh.write(f"{_STATE_LABELS[s].lower()}: {start}-{end}\n")
    print(f"Saved phases   : {phases_path}")

    # ── video ─────────────────────────────────────────────────────────────────
    if not make_video:
        return

    h5_path = data_dir / f"{recording_id}.h5"
    if not h5_path.exists():
        print(f"WARNING: H5 not found ({h5_path}), skipping video.")
        return

    print(f"\nBuilding video from {h5_path} ...")
    video_path = state_dir / f"{recording_id}_object_state.mp4"

    with h5py.File(h5_path, "r") as f:
        rgb_key = "observations/images/aria_rgb_cam/color"
        if rgb_key not in f:
            print(f"WARNING: dataset '{rgb_key}' not in H5, skipping video.")
            return
        rgb_ds = f[rgb_key]
        total  = min(n_frames, rgb_ds.shape[0])
        h_px, w_px = rgb_ds.shape[1], rgb_ds.shape[2]

        import imageio
        writer = imageio.get_writer(str(video_path), fps=fps, codec="libx264",
                                    output_params=["-crf", "18", "-pix_fmt", "yuv420p"])

        for t in range(total):
            if t % 200 == 0:
                print(f"  frame {t}/{total}")

            frame_rgb = np.array(rgb_ds[t], dtype=np.uint8).copy()
            frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

            ex, ey = eef_sm[t, 0], eef_sm[t, 1]
            _threshold_circle(frame_bgr, ex, ey, pos_thresh, bool(pos_ok[t]))
            _dot(frame_bgr, ex, ey,                              (50,  50, 220), radius=6)  # red
            _dot(frame_bgr, centers_sm[t, 0], centers_sm[t, 1], (50, 220,  50), radius=6)  # green

            cx, cy = centers_sm[t, 0], centers_sm[t, 1]
            _annotate(frame_bgr, t, total, dist[t], motion_diff[t], motion_cen_static[t],
                      bool(pos_ok[t]), bool(motion_ok[t]), int(obj_state[t]),
                      ex, ey, cx, cy, static_thresh)

            writer.append_data(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))

        writer.close()

    print(f"Saved video     : {video_path}")


if __name__ == "__main__":
    main()
