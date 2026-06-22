import logging
from pxr import UsdPhysics, PhysxSchema, PhysicsSchemaTools

logger = logging.getLogger(__name__)

_HAND_KEYWORDS = frozenset({
    "thumb", "index", "middle", "ring", "pinky",
    "wrist", "hand", "palm", "finger",
})


class CollisionTracker:
    """
    Counts simulation frames where a hand contacts the environment.

    PhysxContactReportAPI is applied ONLY to non-robot (environment) prims —
    the desk, ground plane, objects, etc. This leaves the articulation links
    completely untouched, avoiding the joint-breakage bug.

    When the callback fires, we check which actor is a robot link and whether
    it is a hand link (by path keywords), giving us:
      - right_hand_desk : right hand link touched environment
      - left_hand_desk  : left hand link touched environment

    Two-phase setup — must happen in order:
      1. tracker.apply_apis()   called BEFORE world.reset()
      2. tracker.subscribe()    called AFTER  world.reset()
    Then per frame: tracker.step() after world.step().
    On finish: tracker.teardown(), logger.info(tracker.report(n_frames)).
    """

    def __init__(self, stage, robot_config, desk_prim_paths: list[str] | None = None):
        self.stage = stage
        self.right_root = robot_config.franka_right_path
        self.left_root = robot_config.franka_left_path
        self._desk_paths: list[str] = desk_prim_paths or []

        self.collision_frames: dict[str, int] = {
            "right_hand_desk": 0,
            "left_hand_desk":  0,
        }
        self._frame_flags: dict[str, bool] = {k: False for k in self.collision_frames}
        self._contact_sub = None

    # ------------------------------------------------------------------
    # Phase 1: apply USD APIs before world.reset()
    # ------------------------------------------------------------------

    def apply_apis(self):
        """Attach PhysxContactReportAPI to environment (non-robot) collision prims.
        Call BEFORE world.reset(). Robot articulation links are never touched.

        Isaac Sim workaround: applying PhysxContactReportAPI to articulation link prims
        corrupts the articulation solver (joints lose constraints, links scatter). To avoid
        this, we apply the API only to environment prims and infer the robot side from the
        actor path in the callback. Isaac Lab's ContactSensor abstraction handles this
        correctly and would allow full per-link collision tracking across all robot bodies."""
        count = 0
        for prim in self.stage.Traverse():
            path = str(prim.GetPath())
            if path.startswith(self.right_root) or path.startswith(self.left_root):
                continue  # never touch robot prims
            if not (prim.HasAPI(UsdPhysics.CollisionAPI) or prim.HasAPI(UsdPhysics.RigidBodyAPI)):
                continue
            if not prim.HasAPI(PhysxSchema.PhysxContactReportAPI):
                PhysxSchema.PhysxContactReportAPI.Apply(prim)
            count += 1
        logger.info("[collision] Contact reporting applied to %d environment prims", count)

    # ------------------------------------------------------------------
    # Phase 2: subscribe to events after world.reset()
    # ------------------------------------------------------------------

    def subscribe(self):
        """Subscribe to contact events. Call AFTER world.reset()."""
        from omni.physx import get_physx_simulation_interface
        self._contact_sub = get_physx_simulation_interface().subscribe_contact_report_events(self._on_contact)
        desk_info = ", ".join(self._desk_paths) if self._desk_paths else "any env prim"
        logger.info("[collision] Tracker active — right=%s  left=%s  desk=%s",
                    self.right_root, self.left_root, desk_info)

    def teardown(self):
        self._contact_sub = None

    # ------------------------------------------------------------------
    # Path helpers
    # ------------------------------------------------------------------

    def _side(self, path: str) -> str | None:
        if path.startswith(self.right_root):
            return "right"
        if path.startswith(self.left_root):
            return "left"
        return None

    def _is_hand(self, path: str) -> bool:
        lower = path.lower()
        return any(kw in lower for kw in _HAND_KEYWORDS)

    # ------------------------------------------------------------------
    # Contact callback
    # ------------------------------------------------------------------

    def _on_contact(self, contact_headers, contact_data):
        for header in contact_headers:
            p0 = str(PhysicsSchemaTools.intToSdfPath(header.actor0))
            p1 = str(PhysicsSchemaTools.intToSdfPath(header.actor1))
            s0 = self._side(p0)
            s1 = self._side(p1)

            # One actor is a robot link, the other is an environment prim (has the API)
            for robot_path, robot_side, other_side, env_path in (
                (p0, s0, s1, p1),
                (p1, s1, s0, p0),
            ):
                if robot_side is None or other_side is not None:
                    continue
                if self._desk_paths and not any(env_path.startswith(d) for d in self._desk_paths):
                    continue
                if self._is_hand(robot_path):
                    self._frame_flags[f"{robot_side}_hand_desk"] = True

    # ------------------------------------------------------------------
    # Per-frame accounting
    # ------------------------------------------------------------------

    def step(self):
        for key in self.collision_frames:
            if self._frame_flags[key]:
                self.collision_frames[key] += 1
        for k in self._frame_flags:
            self._frame_flags[k] = False

    # ------------------------------------------------------------------
    # Reporting
    # ------------------------------------------------------------------

    def report(self, n_frames: int) -> str:
        lines = ["[collision] Frame collision summary:"]
        for key, count in self.collision_frames.items():
            pct = 100.0 * count / n_frames if n_frames > 0 else 0.0
            lines.append(f"  {key:<20s}: {count:5d}/{n_frames} frames ({pct:.1f}%)")
        return "\n".join(lines)
