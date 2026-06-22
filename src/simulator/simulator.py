
from isaacsim import SimulationApp
from isaacsim.core.utils.stage import open_stage
from isaacsim.core.api import World
from isaacsim.core.prims import SingleArticulation, XFormPrim
import omni.usd

import logging
import numpy as np
import time
from pathlib import Path
from scipy.spatial.transform import Rotation

from src.object_state import ObjectState, load_object_state
from src.object_pose import load_object_pose_sequence
from src.visualization import (
    EEFVisualizer,
    VisConfig,
    COLOR_AXIS_X,
    COLOR_AXIS_Y,
    COLOR_AXIS_Z,
    COLOR_AXIS_X_FADED,
    ORIENT_LENGTH,
    FRAME_LINE_SIZE,
)

from src.simulator.quat_utils import (
    normalize_quat_wxyz,
    tool_quat_to_urdf,
    wxyz_to_rotation_matrix,
    parse_ori_correction,
    xyzw_to_wxyz,
    wxyz_to_xyzw,
)
from src.simulator.IK_solver import FrankaIKController
from src.config import RobotConfig, SimConfig, SegConfig, RecordingConfig
from src.loaders.h5_loader import load_replay_h5, RobotSetup
from src.simulator.scene import (
    resolve_descendant_prim_path,
    resolve_dof_indices,
    print_articulation_info,
    ARM_JOINT_NAMES,
    HAND_LEFT_JOINT_NAMES,
    HAND_RIGHT_JOINT_NAMES,
    ActiveObject,
)
from src.simulator.segmentation import Segmentor
from src.simulator.recording import SimulationRecorder
from src.simulator.object_trajectory_recorder import ObjectTrajectoryRecorder
from src.simulator.collision_tracker import CollisionTracker

logger = logging.getLogger(__name__)

_ASSETS = Path(__file__).parents[2] / "assets" / "pandaorca_description-main"

FER_LULA_DESCRIPTION_PATH = str(_ASSETS / "lula" / "fer_robot_descriptor.yaml")
FER_URDF_PATH_RIGHT = str(_ASSETS / "urdf" / "fer_orcahand_right_extended.urdf")
FER_URDF_PATH_LEFT  = str(_ASSETS / "urdf" / "fer_orcahand_left_extended.urdf")

VIZ_OFFSET = np.array([0.0, 0.0, 1.0], dtype=np.float32)

# Orca hand palm prim names used as the reference frame for grasped-object placement.
# These follow the actual simulated kinematic chain (Franka + orca wrist joints).
GRASP_HAND_PRIM_NAME = {"right": "right_palm", "left": "left_palm"}
HAND_COLLISION_KEYWORDS = frozenset((
    "thumb", "index", "middle", "ring", "pinky",
    "wrist", "hand", "palm", "finger",
))


class Simulator:
    def __init__(self, app: SimulationApp, stage_path: str | Path, h5_path: str | Path, sim_config: SimConfig | None = None, vis_config: VisConfig | None = None, robot_config: RobotConfig | None = None, seg_config: SegConfig | None = None, recording_config=None):
        self.app = app
        self.stage_path = str(stage_path)
        self.h5_path = Path(h5_path)
        self.sim_config: SimConfig = sim_config if sim_config is not None else SimConfig()
        self.vis_config = vis_config if vis_config is not None else VisConfig(enabled=False)
        self.robot_config: RobotConfig = robot_config if robot_config is not None else RobotConfig()
        self.seg_config: SegConfig = seg_config if seg_config is not None else SegConfig()
        self.recording_config: RecordingConfig = recording_config if recording_config is not None else RecordingConfig()

        self.set_joints: bool = self.sim_config.set_joints
        self.dt: float = 1.0 / self.sim_config.control_hz
        self.recorder = SimulationRecorder(self.recording_config)
        self.object_trajectory_recorder = ObjectTrajectoryRecorder(self.recording_config)

        missing = []
        for label, path in [
            ("Stage (USD scene)",       self.stage_path),
            ("H5 dataset",              self.h5_path),
            ("FER Lula descriptor",     FER_LULA_DESCRIPTION_PATH),
            ("FER URDF (right)",        FER_URDF_PATH_RIGHT),
            ("FER URDF (left)",         FER_URDF_PATH_LEFT),
        ]:
            if not Path(path).exists():
                missing.append(f"  {label}: {path}")

        if missing:
            raise FileNotFoundError("Missing required files:\n" + "\n".join(missing))

        self.replay = load_replay_h5(self.h5_path)

        _ik_common = dict(
            robot_description_path=FER_LULA_DESCRIPTION_PATH,
            ee_frame_name=self.robot_config.ee_frame_name,
            flange_to_eef_offset=np.asarray(self.robot_config.ee_flange_to_eef_offset, dtype=np.float32),
        )
        self.solver: dict[str, FrankaIKController | None] = {
            "right": FrankaIKController(label="right", urdf_path=FER_URDF_PATH_RIGHT, **_ik_common),
            "left": FrankaIKController(label="left", urdf_path=FER_URDF_PATH_LEFT, **_ik_common) if self.replay.structure == RobotSetup.DUAL else None,
        }

        self._resolve_arm_enables()
        self._setup_scene()
        self._object_free_physics_start_frame: int | None = None
        self._load_objects()
        self._precompute_grasp_anchors()
        self._setup_vis()
        
        # Segmentor requires self.stage, which is set by _setup_scene
        self.segmentor = Segmentor(self.stage, self.robot_config, self.seg_config)
        self.segmentor.setup()

        if self.collision_tracker is not None:
            self.collision_tracker.subscribe()

        self.prev_arm: dict[str, np.ndarray] = {}
        self.ik_fail: dict[str, int] = {"right": 0, "left": 0}
        self.eef_pos: dict[str, np.ndarray | None] = {"right": None, "left": None}
        self.eef_quat: dict[str, np.ndarray | None] = {"right": None, "left": None}

    def _setup_scene(self):
        from pxr import UsdPhysics

        open_stage(self.stage_path)
        self.world = World()
        self.stage = omni.usd.get_context().get_stage()

        logger.debug("[scene] Articulation roots in stage:")
        for prim in self.stage.Traverse():
            if prim.HasAPI(UsdPhysics.ArticulationRootAPI):
                logger.debug("[scene]   %s", prim.GetPath())

        self.arm: dict[str, SingleArticulation] = {
            "right": self.world.scene.add(SingleArticulation(self.robot_config.franka_right_path, name="fer_right")),
            "left": self.world.scene.add(SingleArticulation(self.robot_config.franka_left_path, name="fer_left")),
        }

        self.collision_tracker: CollisionTracker | None = None
        if self.sim_config.collision_tracking:
            self.collision_tracker = CollisionTracker(self.stage, self.robot_config, self.sim_config.desk_prim_paths)
            self.collision_tracker.apply_apis()

        self.world.reset()

        label_right, label_left = "RIGHT COMBINED ROBOT", "LEFT COMBINED ROBOT"
        print_articulation_info(self.arm["right"], label_right)
        print_articulation_info(self.arm["left"], label_left)

        self.arm_idx: dict[str, np.ndarray] = {
            "right": resolve_dof_indices(self.arm["right"], ARM_JOINT_NAMES, label_right),
            "left":  resolve_dof_indices(self.arm["left"],  ARM_JOINT_NAMES, label_left),
        }
        self.hand_idx: dict[str, np.ndarray] = {
            "right": resolve_dof_indices(self.arm["right"], HAND_RIGHT_JOINT_NAMES, label_right),
            "left":  resolve_dof_indices(self.arm["left"],  HAND_LEFT_JOINT_NAMES,  label_left),
        }

    def _resolve_arm_enables(self):
        """
        Set self.enable from H5 structure defaults, overridden by sim_config.
        """
        if self.replay.structure == RobotSetup.MONO:
            h5_right, h5_left = True, False
        else:
            h5_right, h5_left = True, True

        self.enable: dict[str, bool] = {
            "right": self.sim_config.enable_right if self.sim_config.enable_right is not None else h5_right,
            "left": self.sim_config.enable_left if self.sim_config.enable_left is not None else h5_left,
        }

        if self.replay.structure == RobotSetup.MONO and self.enable["left"]:
            logger.warning("[init] left arm enabled but MONO H5 has no dedicated left data — left will mirror right arm")
        if self.replay.structure == RobotSetup.MONO and not self.enable["right"]:
            logger.warning("[init] right arm disabled but it is the only arm with data in MONO H5")

        logger.info("[init] enable_right=%s  enable_left=%s  structure=%s",
                    self.enable["right"], self.enable["left"], self.replay.structure)

    def _load_object_state(self, path: Path) -> np.ndarray | None:
        """Load object state from a .npy or .csv file. Returns (N,) uint8 array or None."""
        if not path.exists():
            logger.warning("[objects] object_state file not found: %s", path)
            return None

        if path.suffix == ".npy":
            raw = np.load(path)  # (N, 2): col0=frame_idx, col1=state
            state = raw[:, 1].astype(np.uint8)
        elif path.suffix == ".csv":
            import csv
            _STATE_MAP = {
                "static":          ObjectState.STATIC,
                "moving":          ObjectState.MOVING,
                "grasped":         ObjectState.GRASPED,
                "occluded":        ObjectState.STATIC,
                "occluded_moving": ObjectState.GRASPED,
            }
            with open(path, newline="") as f:
                reader = csv.DictReader(f)
                rows = list(reader)
            if not rows:
                logger.warning("[objects] object_state CSV is empty: %s", path)
                return None
            frame_key = "stem" if "stem" in rows[0] else "frame" if "frame" in rows[0] else None
            if frame_key is None:
                raise ValueError(f"object_state CSV must contain a 'stem' or 'frame' column: {path}")
            n_frames = max(int(r[frame_key]) for r in rows) + 1
            state = np.zeros(n_frames, dtype=np.uint8)
            for r in rows:
                state[int(r[frame_key])] = int(_STATE_MAP.get(r["state"].strip().lower(), ObjectState.STATIC))
        else:
            logger.warning("[objects] unsupported object_state format: %s", path.suffix)
            return None

        logger.info("[objects] loaded object_state from %s  static=%d  moving=%d  grasped=%d",
                    path.name,
                    (state == ObjectState.STATIC).sum(),
                    (state == ObjectState.MOVING).sum(),
                    (state == ObjectState.GRASPED).sum())
        return state

    def _load_objects(self):
        """
        Load USD objects and their .npy trajectories into the stage, clipping n_frames to the shortest trajectory.
        """
        self.active_objects = []
        self.n_frames = self.replay.n_frames
        self._object_free_physics_start_frame = None
        if not self.sim_config.object_replay:
            logger.info("[objects] object_replay disabled — skipping")
            return
        extra_physics_frames_raw = getattr(self.sim_config, "object_free_physics_extra_frames", 0)
        extra_physics_frames_auto = (
            isinstance(extra_physics_frames_raw, str)
            and extra_physics_frames_raw.strip().lower() == "auto"
        )
        extra_physics_frames = 0 if extra_physics_frames_auto else int(extra_physics_frames_raw or 0)
        if extra_physics_frames < 0:
            raise ValueError("sim.object_free_physics_extra_frames must be >= 0 or 'auto'")
        object_cam = self.sim_config.object_cam
        object_scale = self.sim_config.object_scale
        self.T_cam_world = np.asarray(
            self.robot_config.T_cam_left_world if object_cam == "left" else self.robot_config.T_cam_right_world,
            dtype=np.float64,
        )
        scale_3 = (object_scale, object_scale, object_scale)
        shortest_dynamic_trajectory: int | None = None

        if not self.sim_config.objects:
            logger.info("[objects] no objects configured")
        for obj_cfg in self.sim_config.objects:
            usd_path = Path(obj_cfg.usd_path)
            traj_path = Path(obj_cfg.trajectory_npy)
            prim_path = obj_cfg.prim_path or f"/World/{usd_path.stem}"
            if not usd_path.exists():
                logger.warning("[objects] USD not found: %s — skipping", usd_path)
                continue
            if not traj_path.exists():
                logger.warning("[objects] Trajectory not found: %s — skipping", traj_path)
                continue
            traj = load_object_pose_sequence(traj_path, self.replay.n_frames)
            if not traj.is_static and len(traj) != self.replay.n_frames:
                logger.warning("[objects] %s trajectory length %d != H5 length %d",
                               usd_path.name, len(traj), self.replay.n_frames)
                shortest_dynamic_trajectory = (
                    len(traj)
                    if shortest_dynamic_trajectory is None
                    else min(shortest_dynamic_trajectory, len(traj))
                )
                if extra_physics_frames <= 0:
                    self.n_frames = min(self.n_frames, len(traj))
            self.stage.DefinePrim(prim_path, "Xform").GetReferences().AddReference(str(usd_path))
            prim = XFormPrim(prim_path)
            prim.set_local_scales(np.array([scale_3], dtype=np.float32))
            obj_state = None
            if obj_cfg.object_state is not None:
                obj_state = self._load_object_state(Path(obj_cfg.object_state))

            self.active_objects.append(ActiveObject(
                name=prim_path.rstrip("/").split("/")[-1],
                prim_path=prim_path,
                prim=prim,
                asset_path=usd_path,
                traj=traj,
                state=obj_state,
                grasp_eef_side=obj_cfg.grasp_eef_side,
                grasp_eef_offset=np.asarray(obj_cfg.grasp_eef_offset, dtype=np.float32),
                grasp_ori_correction=parse_ori_correction(obj_cfg.grasp_eef_ori_flip),
                position_offset_world=np.zeros(3, dtype=np.float32),
            ))
            logger.info("[objects] %s  usd=%s  poses=%s  len=%d  scale=%s  frame=%s",
                        prim_path, usd_path.name,
                        "static" if traj.is_static else "trajectory",
                        len(traj), object_scale, self.sim_config.object_trajectory_frame)

        if self.sim_config.object_disable_hand_collision:
            for obj in self.active_objects:
                self._apply_object_physics(obj)

        if extra_physics_frames > 0 and shortest_dynamic_trajectory is not None:
            self._object_free_physics_start_frame = shortest_dynamic_trajectory
            self.n_frames = min(self.replay.n_frames, shortest_dynamic_trajectory + extra_physics_frames)
            logger.info(
                "[objects] free-physics tail enabled  start_frame=%d  extra_frames=%d  replay_frames=%d",
                self._object_free_physics_start_frame,
                extra_physics_frames,
                self.n_frames,
            )
            for obj in self.active_objects:
                self._apply_object_physics(obj)
        elif extra_physics_frames_auto and shortest_dynamic_trajectory is not None:
            self._object_free_physics_start_frame = shortest_dynamic_trajectory
            self.n_frames = self.replay.n_frames
            extra_physics_frames = max(0, self.replay.n_frames - shortest_dynamic_trajectory)
            logger.info(
                "[objects] free-physics tail enabled  start_frame=%d  extra_frames=auto(%d)  replay_frames=%d",
                self._object_free_physics_start_frame,
                extra_physics_frames,
                self.n_frames,
            )
            for obj in self.active_objects:
                self._apply_object_physics(obj)

    def _precompute_grasp_anchors(self):
        """Find first/last grasped frames and apply physics APIs for grasp-anchored objects."""
        if not self.sim_config.object_grasp_anchor or not self.active_objects:
            return
        logger.info("[anchor] grasp anchor enabled — applying physics to %d object(s)", len(self.active_objects))
        for obj in self.active_objects:
            if obj.state is None:
                logger.warning("[anchor] no object_state — grasp anchor skipped for %s", obj.prim.prims[0].GetPath())
                continue
            grasped = np.where(obj.state == ObjectState.GRASPED)[0]
            if len(grasped) == 0:
                logger.warning("[anchor] no GRASPED frames — grasp anchor skipped for %s", obj.prim.prims[0].GetPath())
                continue
            obj.first_grasped_frame = int(grasped[0])
            obj.last_grasped_frame = int(grasped[-1])
            logger.info("[anchor] %s  first_grasped=%d  last_grasped=%d",
                        obj.prim.prims[0].GetPath(), obj.first_grasped_frame, obj.last_grasped_frame)
            self._apply_object_physics(obj)

    def _apply_object_physics(self, obj: "ActiveObject"):
        """Apply RigidBodyAPI (kinematic) and mesh CollisionAPI to an object prim."""
        from pxr import UsdPhysics, UsdGeom, PhysxSchema, Usd

        usd_prim = obj.prim.prims[0]

        rb_api = UsdPhysics.RigidBodyAPI.Apply(usd_prim)
        rb_api.GetKinematicEnabledAttr().Set(True)
        UsdPhysics.MassAPI.Apply(usd_prim).GetMassAttr().Set(0.1)

        physx_rb = PhysxSchema.PhysxRigidBodyAPI.Apply(usd_prim)
        physx_rb.GetLinearDampingAttr().Set(0.2)
        physx_rb.GetAngularDampingAttr().Set(0.2)

        mesh_paths = []
        for child in Usd.PrimRange(usd_prim):
            if child.IsA(UsdGeom.Mesh):
                UsdPhysics.CollisionAPI.Apply(child)
                mesh_api = UsdPhysics.MeshCollisionAPI.Apply(child)
                mesh_api.GetApproximationAttr().Set("convexDecomposition")
                mesh_paths.append(str(child.GetPath()))
        obj.collision_mesh_paths = mesh_paths
        self._apply_object_hand_collision_filter(obj)

        logger.info("[anchor] physics applied to %s (%d mesh(es))", usd_prim.GetPath(), len(mesh_paths))

    def _prim_has_collision_or_body(self, prim) -> bool:
        from pxr import UsdPhysics

        return prim.HasAPI(UsdPhysics.CollisionAPI) or prim.HasAPI(UsdPhysics.RigidBodyAPI)

    def _collision_filter_targets_under(self, root_path: str) -> list[str]:
        from pxr import Usd

        root = self.stage.GetPrimAtPath(root_path)
        if not root.IsValid():
            logger.warning("[collision-filter] prim not found: %s", root_path)
            return []

        targets = []
        for prim in Usd.PrimRange(root):
            if self._prim_has_collision_or_body(prim):
                targets.append(str(prim.GetPath()))
        if not targets:
            targets.append(root_path)
        return targets

    def _auto_hand_collision_filter_targets(self) -> list[str]:
        robot_roots = [self.robot_config.franka_right_path, self.robot_config.franka_left_path]
        targets = []
        for prim in self.stage.Traverse():
            path = str(prim.GetPath())
            if not any(path.startswith(root) for root in robot_roots):
                continue
            lower = path.lower()
            if not any(keyword in lower for keyword in HAND_COLLISION_KEYWORDS):
                continue
            if self._prim_has_collision_or_body(prim):
                targets.append(path)
        return targets

    def _object_hand_collision_filter_targets(self) -> list[str]:
        explicit_paths = list(getattr(self.sim_config, "object_hand_collision_filter_prim_paths", []) or [])
        if explicit_paths:
            targets = []
            for path in explicit_paths:
                targets.extend(self._collision_filter_targets_under(path))
        else:
            targets = self._auto_hand_collision_filter_targets()

        return sorted(set(targets))

    def _apply_object_hand_collision_filter(self, obj: "ActiveObject"):
        if not getattr(self.sim_config, "object_disable_hand_collision", False):
            return

        from pxr import Sdf, UsdPhysics

        targets = self._object_hand_collision_filter_targets()
        if not targets:
            logger.warning("[collision-filter] no hand collision prims found; object-hand collisions remain enabled")
            return

        applied = 0
        for mesh_path in (obj.collision_mesh_paths or []):
            mesh_prim = self.stage.GetPrimAtPath(mesh_path)
            if not mesh_prim.IsValid():
                continue
            rel = UsdPhysics.FilteredPairsAPI.Apply(mesh_prim).GetFilteredPairsRel()
            for target_path in targets:
                rel.AddTarget(Sdf.Path(target_path))
                applied += 1

        logger.info(
            "[collision-filter] disabled %s collisions against %d hand prim(s) via %d pair(s)",
            obj.prim.prims[0].GetPath(),
            len(targets),
            applied,
        )

    def _set_object_collision(self, obj: "ActiveObject", enabled: bool):
        from pxr import UsdPhysics
        for path in (obj.collision_mesh_paths or []):
            prim = self.stage.GetPrimAtPath(path)
            if prim.IsValid():
                UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr().Set(enabled)
        logger.debug("[anchor] collision %s for %s (%d mesh(es))",
                     "enabled" if enabled else "disabled",
                     obj.prim.prims[0].GetPath(), len(obj.collision_mesh_paths or []))

    def _release_frame_for_object(self, obj: "ActiveObject") -> int | None:
        if obj.state is None:
            return None
        grasped = np.where(obj.state == ObjectState.GRASPED)[0]
        if len(grasped) == 0:
            return None
        return min(int(grasped[-1]) + 1, self.n_frames - 1)

    def _align_container_under_release(self):
        """Apply a runtime XY shift so the configured container sits under the release hand pose."""
        if not self.sim_config.object_release_container_align:
            return

        target_path = self.sim_config.object_release_container_prim_path
        target_obj = None
        for obj in self.active_objects:
            obj_path = str(obj.prim.prims[0].GetPath())
            if target_path is not None and obj_path == target_path:
                target_obj = obj
                break
        if target_obj is None:
            logger.warning("[container-align] target prim not found: %s", target_path)
            return

        source_obj = None
        release_frame = None
        for obj in self.active_objects:
            if obj is target_obj:
                continue
            release_frame = self._release_frame_for_object(obj)
            if release_frame is not None:
                source_obj = obj
                break
        if source_obj is None:
            logger.warning("[container-align] no grasped source object found")
            return

        logger.info("[container-align] sampling %s hand pose at release frame %d",
                    source_obj.grasp_eef_side, release_frame)
        for frame in range(release_frame + 1):
            for side in ("right", "left"):
                if self.enable[side]:
                    self._step_arm(side, frame, set_joints=True)
            self.world.step(render=False)

        hand_prim = self.hand_palm_prim[source_obj.grasp_eef_side]
        if hand_prim is None:
            logger.warning("[container-align] hand prim unavailable for %s", source_obj.grasp_eef_side)
            return
        hand_pos, _ = hand_prim.get_world_poses()
        hand_xy = hand_pos[0][:2].astype(np.float32)
        extra_xy = np.asarray(self.sim_config.object_release_container_xy_offset, dtype=np.float32).reshape(-1)
        if extra_xy.shape[0] < 2:
            logger.warning("[container-align] object_release_container_xy_offset must contain x,y")
            extra_xy = np.zeros(2, dtype=np.float32)
        desired_xy = hand_xy + extra_xy[:2]

        target_world = (self.T_cam_world @ target_obj.traj[0]).astype(np.float32)
        current_xy = target_world[:2, 3]
        offset = np.zeros(3, dtype=np.float32)
        offset[:2] = desired_xy - current_xy
        target_obj.position_offset_world = offset

        for side in ("right", "left"):
            if self.enable[side]:
                self._step_arm(side, 0, set_joints=True)
        self.world.step(render=False)

        logger.info("[container-align] %s xy %.3f, %.3f -> %.3f, %.3f  offset=[%.3f, %.3f]",
                    target_obj.prim.prims[0].GetPath(),
                    float(current_xy[0]), float(current_xy[1]),
                    float(desired_xy[0]), float(desired_xy[1]),
                    float(offset[0]), float(offset[1]))

    def _compute_world_bounds(self, prim):
        from pxr import Usd, UsdGeom

        purposes = [UsdGeom.Tokens.default_, UsdGeom.Tokens.render, UsdGeom.Tokens.proxy]
        cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), purposes, True)
        box = cache.ComputeWorldBound(prim).ComputeAlignedBox()
        return (
            np.asarray(box.GetMin(), dtype=np.float32),
            np.asarray(box.GetMax(), dtype=np.float32),
        )

    def _load_obj_vertices(self, path: Path) -> np.ndarray | None:
        cache = getattr(self, "_obj_vertex_cache", None)
        if cache is None:
            cache = {}
            self._obj_vertex_cache = cache

        key = Path(path).resolve()
        if key in cache:
            return cache[key]

        vertices = []
        try:
            with open(key, "r", encoding="utf-8", errors="ignore") as f:
                for line in f:
                    if not line.startswith("v "):
                        continue
                    parts = line.split()
                    if len(parts) < 4:
                        continue
                    vertices.append((float(parts[1]), float(parts[2]), float(parts[3])))
        except OSError as exc:
            logger.warning("[surface-align] could not read OBJ vertices from %s: %s", path, exc)
            cache[key] = None
            return None

        if not vertices:
            logger.warning("[surface-align] no OBJ vertices found in %s", path)
            cache[key] = None
            return None

        cache[key] = np.asarray(vertices, dtype=np.float32)
        return cache[key]

    def _object_mesh_world_bounds(
        self,
        obj: "ActiveObject",
        position: np.ndarray,
        orientation_wxyz: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray] | None:
        asset_path = getattr(obj, "asset_path", None)
        if asset_path is None or Path(asset_path).suffix.lower() != ".obj":
            return None

        vertices = self._load_obj_vertices(Path(asset_path))
        if vertices is None:
            return None

        scale = float(getattr(self.sim_config, "object_scale", 1.0) or 1.0)
        rotation = wxyz_to_rotation_matrix(normalize_quat_wxyz(orientation_wxyz))
        world_vertices = position.reshape(1, 3) + (vertices * scale) @ rotation.T
        return (
            world_vertices.min(axis=0).astype(np.float32),
            world_vertices.max(axis=0).astype(np.float32),
        )

    def _initial_surface_paths(self) -> list[str]:
        paths = list(getattr(self.sim_config, "object_initial_surface_prim_paths", []) or [])
        if paths:
            return paths
        return list(getattr(self.sim_config, "desk_prim_paths", []) or [])

    def _initial_surface_z(self) -> float | None:
        surface_paths = self._initial_surface_paths()
        if not surface_paths:
            logger.warning("[surface-align] no surface prims configured; set sim.object_initial_surface_prim_paths or sim.desk_prim_paths")
            return None

        z_values = []
        for path in surface_paths:
            prim = self.stage.GetPrimAtPath(path)
            if not prim.IsValid():
                logger.warning("[surface-align] surface prim not found: %s", path)
                continue
            _, bound_max = self._compute_world_bounds(prim)
            z_values.append(float(bound_max[2]))

        if not z_values:
            logger.warning("[surface-align] no valid surface prim bounds found")
            return None
        return max(z_values)

    def _surface_aligned_position(
        self,
        obj: "ActiveObject",
        position: np.ndarray,
        orientation_wxyz: np.ndarray,
        surface_z: float,
    ) -> np.ndarray | None:
        clearance = float(getattr(self.sim_config, "object_initial_surface_clearance", 0.0) or 0.0)
        position = position.astype(np.float32).copy()
        orientation_wxyz = orientation_wxyz.astype(np.float32)

        obj.prim.set_world_poses(
            positions=position.reshape(1, 3),
            orientations=orientation_wxyz.reshape(1, 4),
        )
        mesh_bounds = self._object_mesh_world_bounds(obj, position, orientation_wxyz)
        if mesh_bounds is not None:
            bound_min, _ = mesh_bounds
        else:
            bound_min, _ = self._compute_world_bounds(obj.prim.prims[0])
        dz = (surface_z + clearance) - float(bound_min[2])
        position[2] += dz
        return position

    def _align_initial_objects_to_surface(self):
        """Shift configured objects in Z so their frame-0 oriented bounds touch the table surface."""
        if not getattr(self.sim_config, "object_initial_surface_align", False):
            return

        surface_z = self._initial_surface_z()
        if surface_z is None:
            return

        for obj in self.active_objects:
            if len(obj.traj) == 0:
                continue
            first_world = (self.T_cam_world @ obj.traj[0]).astype(np.float32)
            base_pos = first_world[:3, 3].astype(np.float32)
            if obj.position_offset_world is not None:
                base_pos = base_pos + obj.position_offset_world
            base_quat = xyzw_to_wxyz(Rotation.from_matrix(first_world[:3, :3]).as_quat()).astype(np.float32)
            aligned_pos = self._surface_aligned_position(obj, base_pos, base_quat, surface_z)
            if aligned_pos is None:
                continue

            dz = float(aligned_pos[2] - base_pos[2])
            if obj.position_offset_world is None:
                obj.position_offset_world = np.zeros(3, dtype=np.float32)
            obj.position_offset_world = obj.position_offset_world + np.array([0.0, 0.0, dz], dtype=np.float32)
            obj.prim.set_world_poses(
                positions=aligned_pos.reshape(1, 3),
                orientations=base_quat.reshape(1, 4),
            )
            logger.info(
                "[surface-align] %s frame-0 dz=%+.4f surface_z=%.4f",
                obj.prim.prims[0].GetPath(),
                dz,
                surface_z,
            )

    def _settle_object(self, obj: "ActiveObject") -> np.ndarray:
        """
        Fast-forward arms render=False to first_grasped_frame to sample the actual hand palm
        position. Uses hand_pos + R_hand @ grasp_eef_offset as x,y so the object sits exactly
        where the hand will pick it up. Then drops the object to find settled z via physics.
        Sets obj.pre_grasp_quat from the frame just before grasp begins.
        Object is left with rigid body disabled and collision off after settling.
        """
        from pxr import UsdPhysics

        # Step arms forward render=False to first_grasped_frame to sample the actual hand pos
        logger.info("[anchor] fast-forwarding %d steps to sample hand position ...",
                    obj.first_grasped_frame)
        for frame in range(obj.first_grasped_frame + 1):
            for s in ("right", "left"):
                if self.enable[s]:
                    self._step_arm(s, frame, set_joints=True)
            self.world.step(render=False)

        hand_prim = self.hand_palm_prim[obj.grasp_eef_side]
        if hand_prim is not None:
            hand_pos, hand_quat_wxyz_arr = hand_prim.get_world_poses()
            hand_pos = hand_pos[0].astype(np.float32)
            R_hand = wxyz_to_rotation_matrix(hand_quat_wxyz_arr[0].astype(np.float32))
            obj_at_grasp = hand_pos + R_hand @ obj.grasp_eef_offset
            x, y = float(obj_at_grasp[0]), float(obj_at_grasp[1])
            logger.info("[anchor] hand pos=[%.3f, %.3f, %.3f]  object x,y=[%.3f, %.3f]",
                        *hand_pos, x, y)
        else:
            # Fallback: use trajectory position
            first_world = (self.T_cam_world @ obj.traj[obj.first_grasped_frame]).astype(np.float32)
            x, y = float(first_world[0, 3]), float(first_world[1, 3])
            logger.info("[anchor] hand prim unavailable — using trajectory x,y=[%.3f, %.3f]", x, y)

        # Reset both arms to frame 0 so the main play loop warm-starts correctly
        for s in ("right", "left"):
            if self.enable[s]:
                self._step_arm(s, 0, set_joints=True)

        # Orientation from frame just before grasp
        pre_grasp_frame = max(0, obj.first_grasped_frame - 1)
        pre_world = (self.T_cam_world @ obj.traj[pre_grasp_frame]).astype(np.float32)
        obj.pre_grasp_quat = xyzw_to_wxyz(Rotation.from_matrix(pre_world[:3, :3]).as_quat()).astype(np.float32)

        # z_start: trajectory z at first grasped frame + clearance
        first_world = (self.T_cam_world @ obj.traj[obj.first_grasped_frame]).astype(np.float32)
        z_start = float(first_world[2, 3]) + 0.5

        settle_steps = self.sim_config.object_anchor_settle_steps
        logger.info("[anchor] settling %s — dropping from x=%.3f y=%.3f z=%.3f (%d steps) ...",
                    obj.prim.prims[0].GetPath(), x, y, z_start, settle_steps)

        rb_api = UsdPhysics.RigidBodyAPI(obj.prim.prims[0])

        obj.prim.set_world_poses(
            positions=np.array([[x, y, z_start]], dtype=np.float32),
            orientations=obj.pre_grasp_quat.reshape(1, 4),
        )
        rb_api.GetKinematicEnabledAttr().Set(False)

        for _ in range(settle_steps):
            for side in ("right", "left"):
                if self.enable[side] and side in self.prev_arm:
                    q_full = self.arm[side].get_joint_positions().copy()
                    q_full[self.arm_idx[side]] = self.prev_arm[side]
                    self.arm[side].set_joint_positions(q_full)
            self.world.step(render=False)

        pos, _ = obj.prim.get_world_poses()
        settled_z = float(pos[0][2])

        pre_grasp = np.array([x, y, settled_z], dtype=np.float32)
        rb_api.GetKinematicEnabledAttr().Set(True)
        if getattr(self.sim_config, "object_initial_surface_align", False):
            surface_z = self._initial_surface_z()
            if surface_z is not None:
                aligned = self._surface_aligned_position(obj, pre_grasp, obj.pre_grasp_quat, surface_z)
                if aligned is not None:
                    pre_grasp = aligned

        self._set_object_collision(obj, False)  # no collision during pre-grasp / grasped
        obj.prim.set_world_poses(
            positions=pre_grasp.reshape(1, 3),
            orientations=obj.pre_grasp_quat.reshape(1, 4),
        )

        logger.info("[anchor] settled %s → pos=[%.3f, %.3f, %.3f]", obj.prim.prims[0].GetPath(), *pre_grasp)
        return pre_grasp

    def _reset_objects_physics(self):
        """Re-enable kinematic on any objects that were released to physics. Call on reset/seek."""
        from pxr import UsdPhysics
        for obj in self.active_objects:
            obj.grasp_offset_local = None
            obj.grasp_ori_local = None
            obj.grasp_active = False
            obj.physics_release_pending = False
            obj.physics_release_frame = None
            obj.physics_activation_frame = None
            if obj.physics_released:
                self._set_object_collision(obj, False)
                rb_api = UsdPhysics.RigidBodyAPI(obj.prim.prims[0])
                rb_api.GetKinematicEnabledAttr().Set(True)
                rb_api.GetRigidBodyEnabledAttr().Set(True)
                obj.physics_released = False

    def _setup_vis(self):
        """
        Create the EEFVisualizer, compute faded color tuples, and resolve EEF XFormPrims.
        """
        self.visualizer: EEFVisualizer | None = EEFVisualizer() if self.vis_config.enabled else None

        alpha = self.vis_config.eef_alpha if self.vis_config.eef_alpha is not None else (
            0.15 if self.vis_config.video_mode else COLOR_AXIS_X_FADED[3]
        )
        self.cx_f = (COLOR_AXIS_X[0], COLOR_AXIS_X[1], COLOR_AXIS_X[2], alpha)
        self.cy_f = (COLOR_AXIS_Y[0], COLOR_AXIS_Y[1], COLOR_AXIS_Y[2], alpha)
        self.cz_f = (COLOR_AXIS_Z[0], COLOR_AXIS_Z[1], COLOR_AXIS_Z[2], alpha)

        franka_paths = {
            "right": self.robot_config.franka_right_path,
            "left":  self.robot_config.franka_left_path,
        }
        self.robot_base_prim: dict[str, XFormPrim] = {
            side: XFormPrim(path) for side, path in franka_paths.items()
        }
        self.eef_prim: dict[str, XFormPrim | None] = {"right": None, "left": None}
        if self.visualizer is not None:
            self.eef_prim = {
                side: XFormPrim(resolve_descendant_prim_path(self.stage, path, self.robot_config.ee_usd_prim_name))
                for side, path in franka_paths.items()
            }
        self.hand_palm_prim: dict[str, XFormPrim | None] = {
            side: XFormPrim(resolve_descendant_prim_path(self.stage, franka_paths[side], GRASP_HAND_PRIM_NAME[side]))
                  if self.enable[side] else None
            for side in ("right", "left")
        }
        logger.info("[vis] enabled=%s  show_eef=%s  show_offset=%s",
                    self.vis_config.enabled, self.vis_config.show_eef, self.vis_config.show_offset)

    def _step_arm(self, side: str, frame: int, set_joints: bool):
        """
        Run one IK + hand update step for the given arm side ("right" or "left").
        Writes self.eef_pos[side], self.eef_quat[side], self.prev_arm[side], self.ik_fail[side].
        """
        arm_data = getattr(self.replay, f"{side}_arm")
        hand_data = getattr(self.replay, f"{side}_hand")
        arm = self.arm[side]
        solver = self.solver[side]
        arm_idx = self.arm_idx[side]
        hand_idx = self.hand_idx[side]

        wrist_pose = np.asarray(arm_data[frame], dtype=np.float32)
        pos = wrist_pose[:3]
        quat_urdf = tool_quat_to_urdf(normalize_quat_wxyz(wrist_pose[3:7]))
        q_full = arm.get_joint_positions().copy()
        q_arm, ok = solver.compute(
            target_wrist_pos=pos,
            target_quat_wxyz=quat_urdf,
            warm_start=self.prev_arm[side],
        )
        if ok:
            q_full[arm_idx] = q_arm
            self.prev_arm[side] = q_arm.copy()
        else:
            q_full[arm_idx] = self.prev_arm[side]
            self.ik_fail[side] += 1
            logger.warning("[arm] frame %d IK failed %s", frame, side.upper())
        if hand_data is not None:
            q_hand = np.asarray(hand_data[frame], dtype=np.float32).reshape(-1)
            if q_hand.shape[0] == hand_idx.shape[0]:
                q_full[hand_idx] = q_hand
            else:
                logger.warning("[arm] %s hand qpos size %d != expected %d",
                               side.upper(), q_hand.shape[0], hand_idx.shape[0])
        if set_joints:
            arm.set_joint_positions(q_full)

        self.eef_pos[side] = pos
        self.eef_quat[side] = quat_urdf

    def _eef_world_pos(self, side: str) -> np.ndarray | None:
        """Transform H5 EEF position from robot base frame to world frame."""
        pos = self.eef_pos[side]
        if pos is None:
            return None
        base_pos_b, base_quat_b = self.robot_base_prim[side].get_world_poses()
        base_pos = np.array(base_pos_b[0], dtype=np.float32)
        R_base = wxyz_to_rotation_matrix(np.array(base_quat_b[0], dtype=np.float32))
        return R_base @ pos + base_pos

    def _draw_eef(self, side: str):
        """Draw IK target and actual EEF frames for one arm side."""
        act_pos, act_quat_wxyz = self.eef_prim[side].get_world_poses()
        act_quat_xyzw = wxyz_to_xyzw(act_quat_wxyz[0])
        quat_urdf_xyzw = wxyz_to_xyzw(self.eef_quat[side])
        if self.vis_config.show_eef:
            self.visualizer.draw_frame(self.eef_pos[side], quat_urdf_xyzw, COLOR_AXIS_X, COLOR_AXIS_Y, COLOR_AXIS_Z)
            self.visualizer.draw_frame(act_pos[0], act_quat_xyzw, self.cx_f, self.cy_f, self.cz_f)
        if self.vis_config.show_offset:
            self.visualizer.draw_frame(
                self.eef_pos[side] + VIZ_OFFSET,
                quat_urdf_xyzw,
                COLOR_AXIS_X, COLOR_AXIS_Y, COLOR_AXIS_Z,
                length=ORIENT_LENGTH * 0.5,
                width=FRAME_LINE_SIZE * 2,
            )
            self.visualizer.draw_frame(act_pos[0] + VIZ_OFFSET, act_quat_xyzw, self.cx_f, self.cy_f, self.cz_f)

    def _capture_grasp_transform(self, obj: "ActiveObject", hand_pos: np.ndarray, R_hand: np.ndarray):
        """Capture the current object pose relative to the hand so grasp entry is continuous."""
        if obj.pre_grasp_pos is not None and obj.pre_grasp_quat is not None:
            obj_pos = obj.pre_grasp_pos.astype(np.float32)
            R_obj = wxyz_to_rotation_matrix(obj.pre_grasp_quat.astype(np.float32))
        else:
            obj_pos_arr, obj_quat_arr = obj.prim.get_world_poses()
            obj_pos = obj_pos_arr[0].astype(np.float32)
            R_obj = wxyz_to_rotation_matrix(obj_quat_arr[0].astype(np.float32))

        obj.grasp_offset_local = (R_hand.T @ (obj_pos - hand_pos)).astype(np.float32)
        obj.grasp_ori_local = (R_hand.T @ R_obj).astype(np.float32)
        obj.grasp_active = True
        logger.info("[grasp] captured offset for %s: [%.3f, %.3f, %.3f]",
                    obj.prim.prims[0].GetPath(), *obj.grasp_offset_local)

    def _zero_object_velocity(self, obj: "ActiveObject"):
        from pxr import Gf, UsdPhysics

        rb_api = UsdPhysics.RigidBodyAPI(obj.prim.prims[0])
        rb_api.GetVelocityAttr().Set(Gf.Vec3f(0.0, 0.0, 0.0))
        rb_api.GetAngularVelocityAttr().Set(Gf.Vec3f(0.0, 0.0, 0.0))

    def _finish_object_physics_release(self, obj: "ActiveObject"):
        """Enable object collisions and dynamic physics after the release grace period."""
        from pxr import UsdPhysics

        self._set_object_collision(obj, True)
        rb_api = UsdPhysics.RigidBodyAPI(obj.prim.prims[0])
        rb_api.GetRigidBodyEnabledAttr().Set(True)
        rb_api.GetKinematicEnabledAttr().Set(False)
        self._zero_object_velocity(obj)
        obj.physics_released = True
        obj.physics_release_pending = False
        obj.grasp_active = False
        logger.info("[grasp] released %s to physics", obj.prim.prims[0].GetPath())

    def _release_object_to_physics(self, obj: "ActiveObject", frame: int | None = None):
        """Detach a grasped object, then switch it to dynamic physics after a short grace period."""
        if obj.physics_released:
            return
        from pxr import UsdPhysics

        if obj.physics_release_pending:
            if frame is None or frame >= (obj.physics_activation_frame or 0):
                self._finish_object_physics_release(obj)
            return

        delay_frames = int(getattr(self.sim_config, "object_release_collision_delay_frames", 0) or 0)
        if delay_frames < 0:
            raise ValueError("sim.object_release_collision_delay_frames must be >= 0")

        self._set_object_collision(obj, False)
        rb_api = UsdPhysics.RigidBodyAPI(obj.prim.prims[0])
        rb_api.GetRigidBodyEnabledAttr().Set(True)
        rb_api.GetKinematicEnabledAttr().Set(True)
        self._zero_object_velocity(obj)
        obj.grasp_active = False
        obj.physics_release_pending = True
        obj.physics_release_frame = frame
        obj.physics_activation_frame = None if frame is None else frame + delay_frames

        if frame is None or delay_frames == 0:
            self._finish_object_physics_release(obj)
            return

        logger.info(
            "[grasp] detached %s; enabling physics/collision at frame %d",
            obj.prim.prims[0].GetPath(),
            obj.physics_activation_frame,
        )

    def _step_objects(self, frame: int):
        for obj in self.active_objects:
            if (
                self._object_free_physics_start_frame is not None
                and frame >= self._object_free_physics_start_frame
            ):
                self._release_object_to_physics(obj, frame)
                continue
            if frame >= len(obj.traj):
                self._release_object_to_physics(obj, frame)
                continue

            # Recorded world trajectories are final simulation poses. Replay them
            # verbatim, bypassing all camera conversion and state-based rebuilding.
            if self.sim_config.object_trajectory_frame == "world":
                ob_in_world = obj.traj[frame]
                t = ob_in_world[:3, 3].astype(np.float32)
                R_obj = ob_in_world[:3, :3].astype(np.float32)
                q_wxyz = xyzw_to_wxyz(Rotation.from_matrix(R_obj).as_quat())
                obj.prim.set_world_poses(
                    positions=t.reshape(1, 3),
                    orientations=q_wxyz.reshape(1, 4),
                )
                continue

            is_grasped = (
                obj.state is not None
                and frame < len(obj.state)
                and obj.state[frame] == ObjectState.GRASPED
            )

            # --- Grasp-anchor 3-phase logic ---
            if obj.pre_grasp_pos is not None:
                if frame < obj.first_grasped_frame:
                    # Phase 1: hold at settled pre-grasp position with orientation from frame before grasp
                    obj.prim.set_world_poses(
                        positions=obj.pre_grasp_pos.reshape(1, 3),
                        orientations=obj.pre_grasp_quat.reshape(1, 4),
                    )
                    continue
                if obj.physics_released:
                    continue
                if obj.physics_release_pending:
                    self._release_object_to_physics(obj, frame)
                    continue
                elif frame >= obj.first_grasped_frame and not is_grasped:
                    # Phase 3: hand off to physics on the first non-GRASPED frame, then leave alone
                    self._release_object_to_physics(obj, frame)
                    continue
                # Phase 2 (first_grasped_frame <= frame <= last_grasped_frame): fall through to standard logic

            # --- Standard trajectory / palm-attach logic ---
            ob_in_world = self.T_cam_world @ obj.traj[frame]
            t = ob_in_world[:3, 3].astype(np.float32)
            R_obj = ob_in_world[:3, :3].astype(np.float32)
            if obj.position_offset_world is not None:
                t = t + obj.position_offset_world

            if self.sim_config.object_grasp_attach and is_grasped:
                hand_prim = self.hand_palm_prim[obj.grasp_eef_side]
                if hand_prim is not None:
                    hand_pos, hand_quat_wxyz = hand_prim.get_world_poses()
                    hand_pos = hand_pos[0].astype(np.float32)
                    hand_quat_wxyz = hand_quat_wxyz[0].astype(np.float32)
                    R_hand = wxyz_to_rotation_matrix(hand_quat_wxyz)
                    if not obj.grasp_active:
                        self._capture_grasp_transform(obj, hand_pos, R_hand)
                    t = hand_pos + R_hand @ obj.grasp_offset_local
                    R_obj = R_hand @ obj.grasp_ori_local

            q_wxyz = xyzw_to_wxyz(Rotation.from_matrix(R_obj).as_quat())
            obj.prim.set_world_poses(positions=t.reshape(1, 3), orientations=q_wxyz.reshape(1, 4))

    def _log_progress(self, frame: int):
        logger.info("[arm] frame %d/%d", frame, self.n_frames)
        if self.enable["right"]:
            pos_r = "[" + ", ".join(f"{v:.3f}" for v in self.eef_pos["right"]) + "]"
            quat_r = "[" + ", ".join(f"{v:.3f}" for v in self.eef_quat["right"]) + "]"
            logger.debug("[arm]   pos_right=%s  quat_right=%s", pos_r, quat_r)
        if self.enable["left"]:
            pos_l = "[" + ", ".join(f"{v:.3f}" for v in self.eef_pos["left"]) + "]"
            quat_l = "[" + ", ".join(f"{v:.3f}" for v in self.eef_quat["left"]) + "]"
            logger.debug("[arm]   pos_left=%s  quat_left=%s", pos_l, quat_l)
        logger.info("[arm]   IK fails: right=%d  left=%d", self.ik_fail["right"], self.ik_fail["left"])

    def play(self):
        logger.debug("[arm] structure=%s  enable_right=%s  enable_left=%s",
                    self.replay.structure, self.enable["right"], self.enable["left"])

        self.prev_arm = {
            side: self.arm[side].get_joint_positions()[self.arm_idx[side]].copy()
            for side in ("right", "left")
        }
        self.ik_fail = {"right": 0, "left": 0}
        self.eef_pos  = {"right": None, "left": None}
        self.eef_quat = {"right": None, "left": None}

        # Playback control state (written by PlaybackUI callbacks, read by the loop)
        self._paused: bool = False
        self._reset_requested: bool = False
        self._seek_frame: int | None = None

        frame = 0

        self.world.play()
        for _ in range(10):
            self.world.step(render=True)

        self._align_container_under_release()
        self._align_initial_objects_to_surface()

        if self.sim_config.object_grasp_anchor:
            for obj in self.active_objects:
                if obj.first_grasped_frame is not None:
                    obj.pre_grasp_pos = self._settle_object(obj)

        cam_eye = self.sim_config.camera_eye
        cam_target = self.sim_config.camera_target
        if cam_eye is not None and cam_target is not None:
            from isaacsim.core.utils.viewports import set_camera_view
            set_camera_view(eye=np.array(cam_eye), target=np.array(cam_target))

        # Build control panel (GUI mode only)
        playback_ui = None
        if not self.sim_config.headless:
            from src.simulator.playback_ui import PlaybackUI
            playback_ui = PlaybackUI(self.n_frames, self)

        self.recorder.start(h5_path=self.h5_path)
        self.object_trajectory_recorder.start(self.active_objects, h5_path=self.h5_path, n_frames=self.n_frames)

        replay_finished = False  # set once when frame first reaches n_frames

        # Single unified loop — keeps running (and UI alive) until Isaac Sim closes
        while self.app.is_running():

            # Auto-pause and run end-of-replay cleanup the first time we reach the last frame
            if frame >= self.n_frames and not replay_finished:
                replay_finished = True
                self._paused = True
                frame = self.n_frames - 1
                for obj in self.active_objects:
                    if (obj.grasp_active or obj.physics_release_pending) and not obj.physics_released:
                        self._release_object_to_physics(obj)
                logger.info("[arm] Replay finished. IK failures — right: %d/%d  left: %d/%d",
                            self.ik_fail["right"], self.n_frames, self.ik_fail["left"], self.n_frames)
                if self.collision_tracker is not None:
                    logger.info(self.collision_tracker.report(frame))
                    self.collision_tracker.teardown()
                self.recorder.stop()
                self.object_trajectory_recorder.stop()

            # --- Process control requests ---
            frame_jumped = False
            if self._reset_requested:
                frame = 0
                self._reset_requested = False
                self._seek_frame = None
                frame_jumped = True
                replay_finished = False
                self._paused = False  # auto-play on reset
                self._reset_objects_physics()
            if self._seek_frame is not None:
                frame = max(0, min(self._seek_frame, self.n_frames - 1))
                self._seek_frame = None
                frame_jumped = True
                self._reset_objects_physics()

            if playback_ui is not None:
                playback_ui.update(frame)

            # When paused and no jump occurred, just keep rendering
            if self._paused and not frame_jumped:
                self.world.step(render=True)
                time.sleep(self.dt)
                continue

            # --- Apply frame (arm, objects, visualizer) ---
            t_deadline = time.perf_counter() + self.dt
            for side in ("right", "left"):
                if self.enable[side]:
                    self._step_arm(side, frame, self.set_joints)

            self._step_objects(frame)

            if self.segmentor.active and not self._paused:
                self.segmentor.capture(frame)

            if self.visualizer is not None:
                self.visualizer.clear()
                for side in ("right", "left"):
                    if self.enable[side]:
                        self._draw_eef(side)

            if frame % 100 == 0:
                self._log_progress(frame)

            self.world.step(render=True)

            # Object trajectory recording follows natural forward playback only.
            # Pause/seek/reset frames are display interactions and must not write trajectory rows.
            if not replay_finished and not self._paused and not frame_jumped:
                self.object_trajectory_recorder.capture(frame, self.active_objects)

            if not self._paused and not replay_finished:
                if self.collision_tracker is not None:
                    self.collision_tracker.step()
                eef_world_pos = (
                    {side: self._eef_world_pos(side) for side in ("right", "left")}
                    if self.recorder.needs_eef_world_pos(frame)
                    else None
                )
                self.recorder.capture(frame, eef_world_pos=eef_world_pos)
                remaining = t_deadline - time.perf_counter()
                if remaining > 0:
                    time.sleep(remaining)
                frame += 1

        if playback_ui is not None:
            playback_ui.destroy()
        self.recorder.stop()
        self.object_trajectory_recorder.stop()
