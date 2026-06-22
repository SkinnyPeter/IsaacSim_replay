"""
WorldTrajectorySimulator — replays pre-computed world-space object trajectories.

Trajectories produced by ObjectTrajectoryRecorder are already in world space
(shape (N, 4, 4)).  The standard Simulator applies T_cam_world on top of each
pose, which would place objects incorrectly.  This subclass overrides
_step_objects to use the poses verbatim and no-ops the alignment / grasp-anchor
setup that is only meaningful for camera-space input.
"""

import logging

import numpy as np
from scipy.spatial.transform import Rotation

from src.simulator.simulator import Simulator
from src.simulator.quat_utils import xyzw_to_wxyz
from src.simulator.world_traj.frame_window import FrameWindow, load_frame_window

logger = logging.getLogger(__name__)


class WorldTrajectorySimulator(Simulator):
    """Simulator that treats trajectory_npy files as world-space 4×4 transforms."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._robot_frame_window = self._load_robot_frame_window()
        self._apply_robot_frame_window()

    def _load_robot_frame_window(self) -> FrameWindow:
        metadata_path = self.h5_path.parent / "about.yaml"
        try:
            window = load_frame_window(metadata_path)
        except Exception as exc:
            logger.warning("[world-traj] ignoring invalid frame metadata %s: %s", metadata_path, exc)
            return FrameWindow()
        if window.start or window.end is not None:
            logger.info(
                "[world-traj] robot H5 frame window start=%d end=%s",
                window.start,
                "none" if window.end is None else window.end,
            )
        return window

    def _apply_robot_frame_window(self):
        start = self._robot_frame_window.start
        if start >= self.replay.n_frames:
            logger.warning(
                "[world-traj] frame_start=%d is outside H5 length %d; using unshifted robot replay",
                start,
                self.replay.n_frames,
            )
            self._robot_frame_window = FrameWindow()
            return

        available = self.replay.n_frames - start
        if self._robot_frame_window.length is not None:
            available = min(available, self._robot_frame_window.length)
        if available < self.n_frames:
            logger.info(
                "[world-traj] clipping replay length from %d to %d for robot frame window",
                self.n_frames,
                available,
            )
            self.n_frames = available

    # --- no-ops for logic that only applies to camera-space input ---

    def _precompute_grasp_anchors(self):
        pass

    def _align_container_under_release(self):
        pass

    def _align_initial_objects_to_surface(self):
        pass

    # --- robot override: object row 0 maps to H5 frame_start ---

    def _step_arm(self, side: str, frame: int, set_joints: bool):
        robot_frame = frame + self._robot_frame_window.start
        return super()._step_arm(side, robot_frame, set_joints)

    # --- core override: skip T_cam_world, place objects directly ---

    def _step_objects(self, frame: int):
        for obj in self.active_objects:
            if frame >= len(obj.traj):
                continue
            world_pose = obj.traj[frame].astype(np.float32)
            t = world_pose[:3, 3]
            R_obj = world_pose[:3, :3]
            q_wxyz = xyzw_to_wxyz(Rotation.from_matrix(R_obj).as_quat())
            obj.prim.set_world_poses(
                positions=t.reshape(1, 3),
                orientations=q_wxyz.reshape(1, 4),
            )
