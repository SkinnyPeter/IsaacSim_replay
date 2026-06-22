"""Standalone trajectory replay for the rubber duck.

Loads a (N, 4, 4) .npy trajectory (ob_in_cam, OpenCV convention),
optionally smooths it, and bakes it as USD keyframes on the duck prim
so you can scrub the timeline or press Play.

Run with:
    ~/isaac-sim/python.sh scripts/replay_trajectories.py
"""

from isaacsim import SimulationApp
HEADLESS = False
simulation_app = SimulationApp({"headless": HEADLESS})

from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation, Slerp
from scipy.ndimage import gaussian_filter1d

import omni.usd
import omni.timeline
from pxr import Gf, UsdGeom, UsdLux

# ── Configuration ─────────────────────────────────────────────────────────────
BASE_DIR       = Path(__file__).resolve().parent.parent
TRAJECTORY_NPY = BASE_DIR / "data" / "trajectories" / "trajectory.npy"
DUCK_USD       = BASE_DIR / "assets" / "objects" / "rubber_duck.usd"
DUCK_PRIM_PATH = "/World/rubber_duck"

TIME_CODES_PER_FRAME = 20
STAGE_FPS            = 20

SMOOTH            = True
JUMP_THRESH_DEG   = 30.0
TRANSLATION_SIGMA = 5.0
ROTATION_WINDOW   = 21
# ─────────────────────────────────────────────────────────────────────────────

# OpenCV (X right, Y down, Z forward) → USD/Isaac Sim (X right, Y up, Z backward)
CV2USD = np.array([
    [1,  0,  0,  0],
    [0, -1,  0,  0],
    [0,  0, -1,  0],
    [0,  0,  0,  1],
], dtype=np.float64)


def smooth_poses(poses):
    """
    Clean up noisy ob_in_cam trajectories (e.g. from FoundationPose).

    Pass 1 — detect frames where rotation jumps > JUMP_THRESH_DEG (outliers /
    symmetry flips) and replace them with SLERP interpolation.
    Pass 2 — Gaussian filter on translations, rolling quaternion average on rotations.
    """
    N = len(poses)
    poses = poses.copy()

    quats = Rotation.from_matrix(poses[:, :3, :3]).as_quat()
    for i in range(1, N):
        if np.dot(quats[i], quats[i - 1]) < 0:
            quats[i] = -quats[i]

    rot_deltas = np.zeros(N)
    for i in range(1, N):
        dR = poses[i - 1, :3, :3].T @ poses[i, :3, :3]
        rot_deltas[i] = np.degrees(Rotation.from_matrix(dR).magnitude())

    bad = rot_deltas > JUMP_THRESH_DEG
    bad_indices = np.where(bad)[0]
    print(f"Outlier frames (>{JUMP_THRESH_DEG:.0f}° jump): {len(bad_indices)}")

    if bad_indices.size:
        good_mask    = ~bad
        good_mask[0] = True
        good_t   = np.where(good_mask)[0].astype(float)
        good_q   = quats[good_mask]
        good_xyz = poses[good_mask, :3, 3]

        slerp = Slerp(good_t, Rotation.from_quat(good_q))
        all_t = np.arange(N, dtype=float)
        quats[bad_indices] = slerp(all_t[bad_indices]).as_quat()
        for ax in range(3):
            poses[bad_indices, ax, 3] = np.interp(
                all_t[bad_indices], good_t, good_xyz[:, ax])

    poses[:, :3, 3] = gaussian_filter1d(poses[:, :3, 3], sigma=TRANSLATION_SIGMA, axis=0)

    half_w = ROTATION_WINDOW // 2
    quats_smooth = quats.copy()
    for i in range(N):
        window = quats[max(0, i - half_w): min(N, i + half_w + 1)]
        avg = window.mean(axis=0)
        quats_smooth[i] = avg / np.linalg.norm(avg)

    poses[:, :3, :3] = Rotation.from_quat(quats_smooth).as_matrix()
    print(f"Smoothing done (σ_t={TRANSLATION_SIGMA}, rot_window={ROTATION_WINDOW})")
    return poses


def setup_stage():
    ctx = omni.usd.get_context()
    ctx.new_stage()
    stage = ctx.get_stage()
    if stage is None:
        raise RuntimeError("Failed to create a new USD stage.")

    stage.DefinePrim("/World", "Xform")
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    stage.SetFramesPerSecond(STAGE_FPS)

    dome = UsdLux.DomeLight.Define(stage, "/World/DomeLight")
    dome.CreateIntensityAttr(500.0)
    return stage


def add_duck(stage):
    prim = stage.DefinePrim(DUCK_PRIM_PATH, "Xform")
    if not DUCK_USD.exists():
        print(f"WARNING: duck USD not found: {DUCK_USD}")
    else:
        prim.GetReferences().AddReference(str(DUCK_USD))
        print(f"Duck added: {DUCK_PRIM_PATH}")
    return prim


def bake_trajectory(stage, prim, poses):
    """
    Bake (N, 4, 4) ob_in_cam poses as USD keyframes.

    Xform op order: [translate, orient, scale]
    Scale (mm→m) is applied first to vertices, then orient, then translate.
    """
    xform = UsdGeom.Xformable(prim)
    xform.ClearXformOpOrder()

    translate_op = xform.AddTranslateOp(opSuffix="anim")
    orient_op    = xform.AddOrientOp(opSuffix="anim")
    xform.AddScaleOp(opSuffix="init").Set(Gf.Vec3f(0.001, 0.001, 0.001))

    for i, ob_in_cam in enumerate(poses):
        ob_in_usd = CV2USD @ ob_in_cam
        t = ob_in_usd[:3, 3]
        q = Rotation.from_matrix(ob_in_usd[:3, :3]).as_quat()  # xyzw
        qw, qx, qy, qz = float(q[3]), float(q[0]), float(q[1]), float(q[2])

        tc = i * TIME_CODES_PER_FRAME
        translate_op.Set(Gf.Vec3d(float(t[0]), float(t[1]), float(t[2])), tc)
        orient_op.Set(Gf.Quatf(qw, qx, qy, qz), tc)

    print(f"Baked {len(poses)} keyframes onto {prim.GetPath()}")


def main():
    if not TRAJECTORY_NPY.exists():
        raise FileNotFoundError(f"Trajectory not found: {TRAJECTORY_NPY}")

    poses = np.load(str(TRAJECTORY_NPY))
    print(f"Loaded trajectory: {poses.shape}")
    if SMOOTH:
        poses = smooth_poses(poses)

    stage = setup_stage()
    for _ in range(10):
        simulation_app.update()

    duck_prim = add_duck(stage)
    bake_trajectory(stage, duck_prim, poses)

    timeline = omni.timeline.get_timeline_interface()
    last_tc  = (len(poses) - 1) * TIME_CODES_PER_FRAME
    timeline.set_end_time(last_tc / timeline.get_time_codes_per_second())
    timeline.set_current_time(0.0)
    timeline.play()

    print("Playing trajectory — close the window or Ctrl+C to exit.")
    try:
        while simulation_app.is_running():
            simulation_app.update()
    except KeyboardInterrupt:
        pass

    timeline.stop()
    simulation_app.close()


if __name__ == "__main__":
    main()
