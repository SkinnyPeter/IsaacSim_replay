"""
Utilities for recording the Isaac Sim replay.

This module keeps video recording separate from the main Simulator class.
The Simulator should only call:

    recorder.start()
    recorder.capture(frame, eef_world_pos)
    recorder.stop()

The recorder supports:
- one or more cameras recorded in parallel, each to a separate MP4 and/or depth .npy stack
- camera sources: "viewport", "left" (/World/camera_left), "right" (/World/camera_right)
- frame range selection
- direct streaming to MP4, without storing all frames in RAM
- EEF image-space position tracking: .npy + annotated MP4 under EEF_pos/

Output filenames:
  {output_dir}/{id}/mp4/{id}_{camera}.mp4              (rgb_mp4=true)
  {output_dir}/{id}/depth/{camera}/npy/*.npy           (depth_npy=true)
  {output_dir}/{id}/depth/{camera}/png/*.png           (depth_png=true)
  {output_dir}/{id}/EEF_pos/{camera}_{arm}_eef_pos.npy (eef_pos_npy=true)
  {output_dir}/{id}/EEF_pos/{camera}_eef_pos.mp4       (eef_pos_mp4=true)
"""

from datetime import datetime
from pathlib import Path
import logging

from src.config import RecordingConfig

logger = logging.getLogger(__name__)

# Camera names that have a fixed USD prim → supports EEF projection.
_FIXED_CAMERAS = {"left", "right"}


class SimulationRecorder:
    """
    Records RGB frames and/or depth maps from one or more Isaac Sim cameras simultaneously.
    Optionally tracks EEF positions in image space and writes annotated MP4 + .npy files.

    Parameters
    ----------
    config:
        RecordingConfig object.
    """

    def __init__(self, config: RecordingConfig):
        self.config = config

        self._active = False

        # Per-camera state: keyed by camera name.
        self._cam_paths: dict[str, str] = {}
        self._rps: dict = {}
        self._rgb_annotators: dict = {}
        self._depth_annotators: dict = {}
        self._writers: dict = {}
        self._output_paths: dict[str, Path] = {}
        self._depth_dirs: dict[str, Path] = {}
        self._written_frames: dict[str, int] = {}
        self._written_depth_frames: dict[str, int] = {}

        # EEF tracking state.
        self._eef_dir: Path | None = None
        self._cam_intrinsics: dict[str, tuple] = {}   # cam → (fx, fy, cx, cy)
        self._cam_T_world: dict[str, "np.ndarray"] = {}  # cam → 4×4 numpy array (col-vec)
        # cam → arm → list of [frame, u, v]
        self._eef_data: dict[str, dict[str, list]] = {}
        self._eef_writers: dict[str, object] = {}     # cam → imageio writer
        self._eef_output_paths: dict[str, Path] = {}
        # Cameras whose USD transform has not been read yet (deferred until first capture).
        self._eef_setup_pending: set[str] = set()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @property
    def _any_enabled(self) -> bool:
        return self.config.rgb_mp4 or self.config.depth_npy or self.config.depth_png

    @property
    def _any_eef_enabled(self) -> bool:
        return self.config.eef_pos_npy or self.config.eef_pos_mp4

    @property
    def _needs_rgb_annotator(self) -> bool:
        """RGB annotator is needed for clean MP4 or EEF overlay MP4."""
        return self.config.rgb_mp4 or self.config.eef_pos_mp4

    def _setup_eef_camera(self, cam: str, cam_path: str) -> bool:
        """
        Read intrinsics and world pose for *cam* from the live stage.
        Returns False if the prim is missing or lacks camera attributes.
        """
        import numpy as np
        try:
            import omni.usd
            from pxr import UsdGeom, Usd
        except ImportError:
            logger.warning("[recording] EEF tracking: omni.usd/pxr not available")
            return False

        try:
            stage = omni.usd.get_context().get_stage()
            prim = stage.GetPrimAtPath(cam_path)
            if not prim.IsValid():
                logger.warning("[recording] EEF tracking: prim not found at %s", cam_path)
                return False

            cam_usd = UsdGeom.Camera(prim)
            focal_length = cam_usd.GetFocalLengthAttr().Get()
            horiz_ap = cam_usd.GetHorizontalApertureAttr().Get()
            vert_ap = cam_usd.GetVerticalApertureAttr().Get()

            if not focal_length or not horiz_ap or not vert_ap:
                logger.warning("[recording] EEF tracking: missing camera attributes on %s", cam_path)
                return False

            W, H = self.config.resolution
            fx = focal_length * W / horiz_ap
            fy = focal_length * H / vert_ap
            cx = W / 2.0
            cy = H / 2.0
            self._cam_intrinsics[cam] = (fx, fy, cx, cy)

            # USD uses row-vector convention: p_world_row = p_local_row @ M_local_to_world
            # → column-vector world_T_cam = M_local_to_world.T
            xformable = UsdGeom.Xformable(prim)
            local_to_world = xformable.ComputeLocalToWorldTransform(Usd.TimeCode.Default())
            M_raw = np.array(local_to_world, dtype=np.float64)
            if M_raw.shape != (4, 4):
                logger.warning(
                    "[recording] EEF tracking: unexpected GfMatrix4d shape %s on %s", M_raw.shape, cam_path
                )
                return False
            world_T_cam = M_raw.T  # USD row-vector → column-vector convention
            self._cam_T_world[cam] = np.linalg.inv(world_T_cam)

        except Exception:
            logger.exception("[recording] EEF tracking: failed to set up camera %s", cam_path)
            return False

        logger.info(
            "[recording] EEF cam setup  cam=%s  fx=%.1f fy=%.1f  cx=%.1f cy=%.1f",
            cam, fx, fy, cx, cy,
        )
        return True

    def _project_to_image(self, cam: str, pos_world: "np.ndarray") -> tuple[float, float] | None:
        """
        Project a 3-D world-space point onto the image plane of *cam*.

        Returns (u, v) floats, or None if the point is behind the camera.
        USD cameras look along their local -Z axis (X right, Y up in camera frame).
        """
        import numpy as np
        cam_T_world = self._cam_T_world[cam]
        fx, fy, cx, cy = self._cam_intrinsics[cam]

        p_h = np.array([pos_world[0], pos_world[1], pos_world[2], 1.0])
        p_cam = cam_T_world @ p_h          # (4,)

        depth = -p_cam[2]                  # camera looks along -Z
        if depth <= 1e-4:
            return None

        u = fx * p_cam[0] / depth + cx
        v = fy * (-p_cam[1]) / depth + cy  # cam Y is up; image v is down
        return float(u), float(v)

    def _annotate_frame(
        self,
        rgb: "np.ndarray",
        frame: int,
        eef_uvs: "dict[str, tuple[float, float] | None]",
    ) -> "np.ndarray":
        """
        Draw red dot(s) + text overlay on an RGB frame.
        *eef_uvs*: arm → (u, v) or None.
        Returns annotated copy (uint8 H×W×3).
        """
        import numpy as np
        import cv2

        # Convert RGB→BGR for cv2, draw, then convert back to RGB for imageio.
        out = cv2.cvtColor(np.ascontiguousarray(rgb[..., :3], dtype=np.uint8), cv2.COLOR_RGB2BGR)

        arm_colors = {
            "right": (80, 80, 255),    # BGR: red
            "left":  (255, 80, 80),    # BGR: blue
        }

        for i, (arm, uv) in enumerate(eef_uvs.items()):
            color = arm_colors.get(arm, (255, 255, 255))
            if uv is not None:
                u, v = int(round(uv[0])), int(round(uv[1]))
                cv2.circle(out, (u, v), 6, color, -1)
                cv2.circle(out, (u, v), 7, (255, 255, 255), 1)
                label = f"{arm}: ({u},{v})"
                text_y = max(v - 12 - i * 14, 12)
                cv2.putText(out, label, (u + 8, text_y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)

        cv2.putText(out, f"frame {frame}", (6, 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1, cv2.LINE_AA)
        return cv2.cvtColor(out, cv2.COLOR_BGR2RGB)

    def _ensure_eef_writer(self, cam: str):
        if cam in self._eef_writers:
            return
        import imageio
        writer = imageio.get_writer(
            str(self._eef_output_paths[cam]),
            fps=self.config.fps,
            codec="libx264",
            quality=8,
        )
        self._eef_writers[cam] = writer
        logger.info("[recording] EEF video writer initialized  cam=%s  path=%s", cam, self._eef_output_paths[cam])

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def start(self, h5_path=None):
        """
        Initialize the recording backend for all configured cameras.
        """
        if not self.config.enabled or (not self._any_enabled and not self._any_eef_enabled):
            logger.debug("[recording] disabled, skipping start()")
            return

        prefix = Path(h5_path).stem if h5_path is not None else datetime.now().strftime("%Y-%m-%d_%H%M%S")
        output_dir = Path(self.config.output_dir) / prefix

        if self._any_enabled:
            mp4_dir = output_dir / "mp4"
            mp4_dir.mkdir(parents=True, exist_ok=True)

        if self._any_eef_enabled:
            self._eef_dir = output_dir / "EEF_pos"
            self._eef_dir.mkdir(parents=True, exist_ok=True)

        self._active = True
        self._cam_paths.clear()
        self._rps.clear()
        self._rgb_annotators.clear()
        self._depth_annotators.clear()
        self._writers.clear()
        self._output_paths.clear()
        self._depth_dirs.clear()
        self._written_frames.clear()
        self._written_depth_frames.clear()
        self._cam_intrinsics.clear()
        self._cam_T_world.clear()
        self._eef_data.clear()
        self._eef_writers.clear()
        self._eef_output_paths.clear()
        self._eef_setup_pending.clear()

        import omni.replicator.core as rep

        for cam in self.config.cameras:
            if cam == "left":
                cam_path = "/World/camera_left"
            elif cam == "right":
                cam_path = "/World/camera_right"
            elif cam == "viewport":
                import omni.kit.viewport.utility as vp_utils
                viewport = vp_utils.get_active_viewport()
                if viewport is None:
                    raise RuntimeError(
                        "[recording] camera='viewport' requested but no active viewport found. "
                        "Use camera='left' or camera='right' for headless recording."
                    )
                cam_path = str(viewport.get_active_camera())
            else:
                raise ValueError(
                    f"Unknown recording camera '{cam}'. "
                    "Expected one of: 'viewport', 'left', 'right'."
                )

            rp = rep.create.render_product(cam_path, self.config.resolution)

            if self._needs_rgb_annotator:
                annotator = rep.AnnotatorRegistry.get_annotator("rgb")
                annotator.attach([rp])
                self._rgb_annotators[cam] = annotator
                if self.config.rgb_mp4:
                    self._output_paths[cam] = output_dir / "mp4" / f"{prefix}_{cam}.mp4"
                self._written_frames[cam] = 0

            if self.config.depth_npy or self.config.depth_png:
                depth_ann = rep.AnnotatorRegistry.get_annotator("distance_to_camera")
                depth_ann.attach([rp])
                self._depth_annotators[cam] = depth_ann
                depth_dir = output_dir / "depth" / cam
                if self.config.depth_npy:
                    (depth_dir / "npy").mkdir(parents=True, exist_ok=True)
                if self.config.depth_png:
                    (depth_dir / "png").mkdir(parents=True, exist_ok=True)
                self._depth_dirs[cam] = depth_dir
                self._written_depth_frames[cam] = 0

            # EEF tracking: only supported for named fixed cameras.
            # USD transforms are deferred until first capture() so they are read
            # after at least one world.step() has settled the physics scene.
            if self._any_eef_enabled and cam in _FIXED_CAMERAS:
                self._eef_setup_pending.add(cam)
                self._eef_data[cam] = {}
                if self.config.eef_pos_mp4:
                    self._eef_output_paths[cam] = self._eef_dir / f"cam{cam.capitalize()}_eef_pos.mp4"

            self._cam_paths[cam] = cam_path
            self._rps[cam] = rp

            logger.info(
                "[recording] Replicator render product attached  cam=%s  prim=%s  resolution=%s"
                "  rgb_mp4=%s  depth_npy=%s  depth_png=%s  eef_pos_npy=%s  eef_pos_mp4=%s",
                cam, cam_path, self.config.resolution,
                self.config.rgb_mp4, self.config.depth_npy, self.config.depth_png,
                self.config.eef_pos_npy, self.config.eef_pos_mp4,
            )

        logger.info(
            "[recording] started  cameras=%s  output_dir=%s  fps=%d  frames=%s:%s",
            list(self.config.cameras),
            output_dir,
            self.config.fps,
            self.config.start_frame,
            self.config.end_frame,
        )

    def should_record(self, frame: int) -> bool:
        """Return True if the current frame is inside the requested recording range."""
        if not self.config.enabled or not (self._any_enabled or self._any_eef_enabled) or not self._active:
            return False
        if frame < self.config.start_frame:
            return False
        if self.config.end_frame is not None and frame > self.config.end_frame:
            return False
        return True

    def needs_eef_world_pos(self, frame: int) -> bool:
        """Return True if the caller must supply eef_world_pos for this frame."""
        return self._any_eef_enabled and self.should_record(frame)

    def _get_rgb_frame(self, cam: str):
        """Capture one RGB frame; returns (H, W, 3) uint8 ndarray or None."""
        rgb = self._rgb_annotators[cam].get_data()
        if rgb is None or rgb.size == 0:
            return None
        if rgb.ndim == 1:
            w, h = self.config.resolution
            rgb = rgb.reshape(h, w, -1)
        return rgb[..., :3]

    def _get_depth_frame(self, cam: str):
        """Capture one depth frame; returns (H, W) float32 ndarray or None."""
        import numpy as np
        depth = self._depth_annotators[cam].get_data()
        if depth is None:
            return None
        if hasattr(depth, "size") and depth.size == 0:
            return None
        if isinstance(depth, np.ndarray) and depth.ndim == 1:
            w, h = self.config.resolution
            depth = depth.reshape(h, w)
        return depth.astype(np.float32)

    def _ensure_writer(self, cam: str):
        """Lazily open the clean imageio video writer for *cam*."""
        if cam in self._writers:
            return
        import imageio
        writer = imageio.get_writer(
            str(self._output_paths[cam]),
            fps=self.config.fps,
            codec="libx264",
            quality=8,
        )
        self._writers[cam] = writer
        logger.info("[recording] video writer initialized  cam=%s  path=%s", cam, self._output_paths[cam])

    def capture(self, frame: int, eef_world_pos: "dict[str, np.ndarray | None] | None" = None):
        """
        Capture and write one frame per camera if inside the recording range.

        Parameters
        ----------
        frame:
            Simulation frame index.
        eef_world_pos:
            Dict mapping arm name ("right", "left") to 3-D world-space position
            as a numpy array, or None if the arm is inactive.
        """
        if not self.should_record(frame):
            return

        import numpy as np

        # Lazy camera setup: deferred from start() so USD transforms are read
        # after the first world.step() has settled the scene.
        for cam in list(self._eef_setup_pending):
            cam_path = self._cam_paths[cam]
            if self._setup_eef_camera(cam, cam_path):
                self._eef_setup_pending.discard(cam)
                logger.info("[recording] EEF cam setup completed (deferred)  cam=%s", cam)
            else:
                logger.warning("[recording] EEF tracking disabled for cam=%s (setup failed)", cam)
                self._eef_setup_pending.discard(cam)
                del self._eef_data[cam]
                self._eef_output_paths.pop(cam, None)

        for cam in self.config.cameras:
            # Capture RGB once (shared by clean MP4 and EEF annotated MP4).
            rgb = None
            if cam in self._rgb_annotators:
                rgb = self._get_rgb_frame(cam)
                if rgb is None:
                    logger.warning("[recording] frame %d cam=%s: failed to capture RGB", frame, cam)

            # --- clean RGB MP4 ---
            if self.config.rgb_mp4 and rgb is not None:
                self._ensure_writer(cam)
                self._writers[cam].append_data(rgb)
                self._written_frames[cam] = self._written_frames.get(cam, 0) + 1

            # --- depth ---
            if self.config.depth_npy or self.config.depth_png:
                depth = self._get_depth_frame(cam)
                if depth is None:
                    logger.warning("[recording] frame %d cam=%s: failed to capture depth", frame, cam)
                else:
                    if self.config.depth_npy:
                        np.save(self._depth_dirs[cam] / "npy" / f"{frame:06d}.npy", depth)
                    if self.config.depth_png:
                        import cv2
                        depth_mm = np.clip(depth * 1000.0, 0, 65535).astype(np.uint16)
                        cv2.imwrite(str(self._depth_dirs[cam] / "png" / f"{frame:06d}.png"), depth_mm)
                    self._written_depth_frames[cam] = self._written_depth_frames.get(cam, 0) + 1

            # --- EEF tracking ---
            if self._any_eef_enabled and cam in self._eef_data and eef_world_pos is not None:
                eef_uvs: dict[str, tuple[float, float] | None] = {}
                for arm, pos in eef_world_pos.items():
                    if pos is None:
                        eef_uvs[arm] = None
                        continue
                    uv = self._project_to_image(cam, pos)
                    eef_uvs[arm] = uv
                    if self.config.eef_pos_npy:
                        u_val = uv[0] if uv is not None else float("nan")
                        v_val = uv[1] if uv is not None else float("nan")
                        self._eef_data[cam].setdefault(arm, []).append([frame, u_val, v_val])

                if self.config.eef_pos_mp4:
                    if rgb is None:
                        logger.warning(
                            "[recording] frame %d cam=%s: EEF MP4 frame skipped — RGB unavailable "
                            "(MP4 frame count will diverge from .npy row count)", frame, cam
                        )
                    else:
                        annotated = self._annotate_frame(rgb, frame, eef_uvs)
                        self._ensure_eef_writer(cam)
                        self._eef_writers[cam].append_data(annotated)

    def stop(self):
        """Finish recording, save .npy files, and release all resources."""
        if not self._active:
            return

        for writer in self._writers.values():
            writer.close()
        self._writers.clear()

        for writer in self._eef_writers.values():
            writer.close()
        self._eef_writers.clear()

        # Save EEF .npy files.
        if self.config.eef_pos_npy and self._eef_dir is not None:
            import numpy as np
            for cam, arms in self._eef_data.items():
                for arm, rows in arms.items():
                    if not rows:
                        logger.debug(
                            "[recording] EEF npy: cam=%s arm=%s produced no data — skipping", cam, arm
                        )
                        continue
                    arr = np.array(rows, dtype=np.float64)  # (N, 3): [frame, u, v]
                    npy_path = self._eef_dir / f"cam{cam.capitalize()}_{arm}EEF_pos.npy"
                    np.save(npy_path, arr)
                    logger.info(
                        "[recording] EEF npy saved  cam=%s  arm=%s  frames=%d  path=%s",
                        cam, arm, len(rows), npy_path,
                    )

        for rp in self._rps.values():
            # Annotators are implicitly detached when the render product is destroyed.
            rp.destroy()
        self._rps.clear()

        self._rgb_annotators.clear()
        self._depth_annotators.clear()
        self._active = False

        # Summary logs.
        for cam in self.config.cameras:
            if self.config.rgb_mp4:
                n = self._written_frames.get(cam, 0)
                if n == 0:
                    logger.warning(
                        "[recording] cam=%s: no RGB frames written — check frame range or camera config", cam
                    )
                else:
                    logger.info("[recording] cam=%s RGB saved %s (%d frames)", cam, self._output_paths.get(cam), n)

            if self.config.depth_npy or self.config.depth_png:
                n = self._written_depth_frames.get(cam, 0)
                if n == 0:
                    logger.warning(
                        "[recording] cam=%s: no depth frames written — check frame range or camera config", cam
                    )
                else:
                    if self.config.depth_npy:
                        logger.info("[recording] cam=%s depth npy saved %s (%d frames)", cam, self._depth_dirs[cam] / "npy", n)
                    if self.config.depth_png:
                        logger.info("[recording] cam=%s depth png saved %s (%d frames)", cam, self._depth_dirs[cam] / "png", n)

            if self.config.eef_pos_npy and cam in self._eef_data:
                n_arms = sum(1 for rows in self._eef_data[cam].values() if rows)
                if n_arms == 0:
                    logger.warning("[recording] cam=%s: eef_pos_npy enabled but no EEF data recorded", cam)
                else:
                    logger.info("[recording] cam=%s EEF npy saved (%d arm(s))  dir=%s", cam, n_arms, self._eef_dir)

            if self.config.eef_pos_mp4 and cam in self._eef_output_paths:
                logger.info("[recording] cam=%s EEF MP4 saved %s", cam, self._eef_output_paths[cam])
