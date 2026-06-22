"""
Record final simulated object trajectories.

The simulator supplies the active object prims after each simulation step. This
recorder reads the final world pose of each object and saves one trajectory file
per object:

  {output_dir}/{h5_id}/object_trajectory/{object_name}.npy

Each file has shape (N, 4, 4), using world-space homogeneous transforms.
"""

from pathlib import Path
import logging
import re

import numpy as np
from scipy.spatial.transform import Rotation

from src.config import RecordingConfig
from src.simulator.quat_utils import wxyz_to_xyzw

logger = logging.getLogger(__name__)
PROJECT_ROOT = Path(__file__).resolve().parents[2]


class ObjectTrajectoryRecorder:
    """Collect and save world-space object poses from the live stage."""

    def __init__(self, config: RecordingConfig):
        self.config = config
        self._active = False
        self._output_dir: Path | None = None
        self._trajectories: dict[str, np.ndarray] = {}
        self._seen_frames: dict[str, np.ndarray] = {}
        self._object_output_names: dict[str, str] = {}
        self._start_frame = 0
        self._end_frame = -1
        self._last_recorded_frame: int | None = None

    @property
    def enabled(self) -> bool:
        object_config = getattr(self.config, "object_trajectory", None)
        if object_config is not None:
            return bool(object_config.enabled)
        return bool(self.config.object_trajectory_npy)

    @property
    def _configured_output_dir(self) -> str:
        object_config = getattr(self.config, "object_trajectory", None)
        output_dir = getattr(object_config, "output_dir", None) if object_config is not None else None
        return output_dir or self.config.output_dir

    @property
    def _configured_start_frame(self) -> int:
        object_config = getattr(self.config, "object_trajectory", None)
        start_frame = getattr(object_config, "start_frame", None) if object_config is not None else None
        return self.config.start_frame if start_frame is None else int(start_frame)

    @property
    def _configured_end_frame(self) -> int | None:
        object_config = getattr(self.config, "object_trajectory", None)
        end_frame = getattr(object_config, "end_frame", None) if object_config is not None else None
        if object_config is not None:
            return None if end_frame is None else int(end_frame)
        return self.config.end_frame

    def start(self, active_objects, h5_path=None, n_frames: int | None = None):
        if not self.enabled:
            logger.info("[object-traj] disabled; set object_trajectory.enabled=true to save object trajectories")
            return

        if not active_objects:
            logger.warning("[object-traj] enabled but no active objects are loaded")
            return

        prefix = Path(h5_path).stem if h5_path is not None else "simulation"
        output_root = self._resolve_output_root(h5_path)
        self._output_dir = output_root / prefix / "object_trajectory"
        self._output_dir.mkdir(parents=True, exist_ok=True)

        self._start_frame = max(0, self._configured_start_frame)
        configured_end = self._configured_end_frame
        if n_frames is None:
            self._end_frame = configured_end if configured_end is not None else -1
        else:
            max_frame = max(0, int(n_frames) - 1)
            effective_end = max_frame if configured_end is None else configured_end
            self._end_frame = min(max_frame, effective_end)

        frame_count = max(0, self._end_frame - self._start_frame + 1)
        if frame_count == 0:
            logger.warning(
                "[object-traj] enabled but frame range is empty  start=%d  end=%d",
                self._start_frame,
                self._end_frame,
            )
            return

        self._object_output_names = {}
        self._trajectories = {}
        self._seen_frames = {}
        self._last_recorded_frame = self._start_frame - 1
        seen: dict[str, int] = {}
        for obj in active_objects:
            base_name = self._safe_filename(obj.name)
            seen[base_name] = seen.get(base_name, 0) + 1
            output_name = base_name if seen[base_name] == 1 else f"{base_name}_{seen[base_name]}"
            self._object_output_names[obj.prim_path] = output_name
            self._trajectories[output_name] = np.full((frame_count, 4, 4), np.nan, dtype=np.float64)
            self._seen_frames[output_name] = np.zeros(frame_count, dtype=bool)

        self._active = True
        logger.info(
            "[object-traj] started  objects=%s  output_dir=%s  frames=%d:%d",
            list(self._trajectories),
            self._output_dir,
            self._start_frame,
            self._end_frame,
        )

    @staticmethod
    def _safe_filename(name: str) -> str:
        safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", name).strip("._")
        return safe or "object"

    def _resolve_output_root(self, h5_path) -> Path:
        output_root = Path(self._configured_output_dir)
        if output_root.is_absolute():
            return output_root

        return PROJECT_ROOT / output_root

    def should_record(self, frame: int) -> bool:
        if not self._active:
            return False
        return self._start_frame <= frame <= self._end_frame

    def capture(self, frame: int, active_objects):
        if not self.should_record(frame):
            return

        if self._last_recorded_frame is not None:
            if frame <= self._last_recorded_frame:
                logger.debug(
                    "[object-traj] ignoring non-chronological frame %d; last recorded frame is %d",
                    frame,
                    self._last_recorded_frame,
                )
                return
            if frame > self._last_recorded_frame + 1:
                logger.warning(
                    "[object-traj] frame jump detected while recording: expected %d, got %d; missing rows remain NaN",
                    self._last_recorded_frame + 1,
                    frame,
                )

        for obj in active_objects:
            positions, orientations = obj.prim.get_world_poses()
            pos = np.asarray(positions[0], dtype=np.float64)
            quat_wxyz = np.asarray(orientations[0], dtype=np.float64)

            transform = np.eye(4, dtype=np.float64)
            transform[:3, :3] = Rotation.from_quat(wxyz_to_xyzw(quat_wxyz)).as_matrix()
            transform[:3, 3] = pos

            output_name = self._object_output_names.get(obj.prim_path, self._safe_filename(obj.name))
            if output_name not in self._trajectories:
                continue
            row = frame - self._start_frame
            self._trajectories[output_name][row] = transform
            self._seen_frames[output_name][row] = True

        self._last_recorded_frame = frame

    def stop(self):
        if not self._active:
            return

        if self._output_dir is None:
            self._active = False
            return

        saved_paths: list[Path] = []
        for name, arr in self._trajectories.items():
            seen = self._seen_frames.get(name)
            if seen is None or not seen.any():
                logger.warning("[object-traj] %s produced no poses", name)
                continue

            out_path = self._output_dir / f"{name}.npy"
            np.save(out_path, arr)
            saved_paths.append(out_path)
            missing = int((~seen).sum())
            if missing:
                logger.warning(
                    "[object-traj] %s has %d unrecorded frame(s); missing rows are NaN",
                    name,
                    missing,
                )
            logger.info(
                "[object-traj] saved  object=%s  frames=%d  path=%s",
                name,
                arr.shape[0],
                out_path,
            )

        if saved_paths:
            logger.info("[object-traj] complete  saved_files=%d  dir=%s", len(saved_paths), self._output_dir)

        self._active = False
        self._last_recorded_frame = None
