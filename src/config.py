from dataclasses import dataclass, field
from pathlib import Path

from src.object_state import ObjectState  # noqa: F401  — re-exported for convenience


@dataclass
class ObjectConfig:
    usd_path: Path
    trajectory_npy: Path
    prim_path: str | None = None  # auto: /World/<usd_stem> if None
    object_state: Path | None = None  # .npy: (N,2) [frame_idx, state]; .csv: stem,state columns
    grasp_eef_side: str = "right"       # which arm EEF to follow when grasped
    grasp_eef_offset: list = field(default_factory=lambda: [0.0, 0.0, 0.0])  # world-frame offset from EEF to object when grasped
    grasp_eef_ori_flip: str = "none"    # apply 180° rotation around this axis on top of EEF ori: "none", "x", "y", "z"


@dataclass
class RobotConfig:
    franka_right_path: str = "/World/fer_orcahand_right_extended"
    franka_left_path: str = "/World/fer_orcahand_left_extended"
    ee_frame_name: str = "fer_link8"
    ee_usd_prim_name: str = "fer_link8"
    ee_flange_to_eef_offset: list = field(default_factory=lambda: [0.13, 0.0, 0.07])
    T_cam_left_world: list = field(default_factory=lambda: [
        [-0.02199727, -0.80581615,  0.59175708, -0.04596533],
        [-0.99905014,  0.03998766,  0.01731508,  0.09513673],
        [-0.03761575, -0.59081411, -0.80593036,  1.20379187],
        [ 0.0,         0.0,         0.0,         1.0        ],
    ])
    T_cam_right_world: list = field(default_factory=lambda: [
        [ 0.02933941, -0.83227828,  0.55358113, -0.07484866],
        [-0.99642232,  0.01956109,  0.08221870, -0.00350517],
        [-0.07925749, -0.55401284, -0.82872675,  1.23895363],
        [ 0.0,         0.0,         0.0,         1.0        ],
    ])


@dataclass
class LogConfig:
    file_enabled: bool = False
    file_level: str = "DEBUG"     # level written to the log file
    console_level: str = "INFO"   # level written to stdout


@dataclass
class SegConfig:
    enabled: bool = False
    output_dir: str = "data/segmentation"
    save_rate: int = 1
    mask_png: bool = False
    mask_npy: bool = False
    rgb_png: bool = False


@dataclass
class ObjectTrajectoryRecordingConfig:
    enabled: bool = False
    output_dir: str | None = None
    start_frame: int | None = None
    end_frame: int | None = None


@dataclass
class RecordingConfig:
    enabled: bool = True
    output_dir: str = "outputs/"
    cameras: list[str] = field(default_factory=lambda: ["viewport"])  # any of: "viewport", "left", "right"
    fps: int = 30
    start_frame: int = 0
    end_frame: int | None = None
    resolution: tuple[int, int] = (640, 480)
    rgb_mp4: bool = False
    depth_npy: bool = False
    depth_png: bool = False
    eef_pos_npy: bool = False   # save per-arm EEF image-space positions as .npy under EEF_pos/
    eef_pos_mp4: bool = False   # save annotated MP4 with red dot + frame/pos overlay under EEF_pos/
    object_trajectory_npy: bool = False  # legacy: use object_trajectory.enabled for new configs
    object_trajectory: ObjectTrajectoryRecordingConfig = field(default_factory=ObjectTrajectoryRecordingConfig)

@dataclass
class SimConfig:
    headless: bool = False
    set_joints: bool = True
    enable_right: bool | None = None   # None → infer from H5 structure
    enable_left: bool | None = None    # None → infer from H5 structure
    collision_tracking: bool = True
    desk_prim_paths: list[str] = field(default_factory=list)
    camera_eye: tuple | None = (1.97035, 0.00915, 1.58108)
    camera_target: tuple | None = (0.51, 0.0, 1.23)
    object_cam: str = "right"
    object_trajectory_frame: str = "camera"  # "camera" (legacy) or direct "world" poses
    object_scale: float = 0.001
    object_replay: bool = True
    object_grasp_attach: bool = True   # attach object to hand palm during GRASPED frames
    object_grasp_anchor: bool = False  # hold static phases at grasp position; release with physics post-grasp
    object_anchor_settle_steps: int = 120  # render=False physics steps to let object reach resting position
    object_initial_surface_align: bool = False  # shift initial object poses in z so their bounds touch the table surface
    object_initial_surface_prim_paths: list[str] = field(default_factory=list)  # table/surface prims; defaults to desk_prim_paths
    object_initial_surface_clearance: float = 0.0  # extra meters above the detected surface
    object_disable_hand_collision: bool = False  # filter object collisions against hand links while keeping table/object collisions
    object_hand_collision_filter_prim_paths: list[str] = field(default_factory=list)  # explicit hand/filter prims; empty auto-detects
    object_release_collision_delay_frames: int = 0  # optional grace period before enabling released object collision/dynamics
    object_free_physics_extra_frames: int | str = 0  # int or "auto": keep replaying arms while objects run with physics
    object_release_container_align: bool = False  # move a container under the hand at object release
    object_release_container_prim_path: str | None = None
    object_release_container_xy_offset: list = field(default_factory=lambda: [0.0, 0.0])
    control_hz: float = 30
    objects: list = field(default_factory=list)  # list[ObjectConfig]


def _h5_id_from_manipulation_id(manipulation_id) -> str:
    text = str(manipulation_id)
    return text.split("_demo_", 1)[0] if "_demo_" in text else text


def _template_vars(raw: dict) -> dict[str, str]:
    manipulation_id = raw.get("manipulation_id")
    variables = {"manipulation_id": str(manipulation_id)} if manipulation_id is not None else {}
    if raw.get("h5_id") is not None:
        variables["h5_id"] = str(raw["h5_id"])
    elif manipulation_id is not None:
        variables["h5_id"] = _h5_id_from_manipulation_id(manipulation_id)
    return variables


def _expand_template(value, variables: dict[str, str]):
    if isinstance(value, str):
        for key, replacement in variables.items():
            value = value.replace("{" + key + "}", replacement)
    return value


def _expand_templates(value, variables: dict[str, str]):
    if isinstance(value, dict):
        return {key: _expand_templates(item, variables) for key, item in value.items()}
    if isinstance(value, list):
        return [_expand_templates(item, variables) for item in value]
    return _expand_template(value, variables)


def _resolve_path(base: Path, value, variables: dict[str, str]) -> Path:
    path = Path(_expand_template(value, variables))
    return path if path.is_absolute() else base / path


def load_config(yaml_path: Path, base_dir: Path | None = None, manipulation_id: str | None = None) -> tuple:
    """Load config.yaml and return (scene_path, data_path, RobotConfig, SimConfig, VisConfig, SegConfig, LogConfig, RecordingConfig)."""
    import yaml
    from src.visualization import VisConfig

    with open(yaml_path) as f:
        raw = yaml.safe_load(f)

    base = base_dir if base_dir is not None else yaml_path.parent
    variables = _template_vars(raw)
    if manipulation_id is not None:
        variables["manipulation_id"] = str(manipulation_id)
        if raw.get("h5_id") is None:
            variables["h5_id"] = _h5_id_from_manipulation_id(manipulation_id)
    raw = _expand_templates(raw, variables)
    scene_path = _resolve_path(base, raw["scene_path"], variables)
    default_data_path = "data/{manipulation_id}/{h5_id}.h5" if "manipulation_id" in variables else raw["data_path"]
    data_path = _resolve_path(base, raw.get("data_path", default_data_path), variables)

    r = raw.get("robot", {})
    robot = RobotConfig(
        franka_right_path=r.get("franka_right_path", "/World/fer_orcahand_right_extended"),
        franka_left_path=r.get("franka_left_path", "/World/fer_orcahand_left_extended"),
        ee_frame_name=r.get("ee_frame_name", "fer_link8"),
        ee_usd_prim_name=r.get("ee_usd_prim_name", "fer_link8"),
        ee_flange_to_eef_offset=r.get("ee_flange_to_eef_offset", [0.13, 0.0, 0.07]),
        T_cam_left_world=r.get("T_cam_left_world", []),
        T_cam_right_world=r.get("T_cam_right_world", []),
    )

    s = raw.get("sim", {})
    object_trajectory_frame = str(s.get("object_trajectory_frame", "camera")).lower()
    if object_trajectory_frame not in {"camera", "world"}:
        raise ValueError("sim.object_trajectory_frame must be 'camera' or 'world'")
    objects = [
        ObjectConfig(
            usd_path=_resolve_path(base, obj["usd_path"], variables),
            trajectory_npy=_resolve_path(base, obj["trajectory_npy"], variables),
            prim_path=obj.get("prim_path"),
            object_state=_resolve_path(base, obj["object_state"], variables) if "object_state" in obj
                    else _resolve_path(base, obj["object_state_npy"], variables) if "object_state_npy" in obj
                    else None,
            grasp_eef_side=obj.get("grasp_eef_side", "right"),
            grasp_eef_offset=obj.get("grasp_eef_offset", [0.0, 0.0, 0.0]),
            grasp_eef_ori_flip=obj.get("grasp_eef_ori_flip", "none"),
        )
        for obj in s.get("objects", [])
    ]
    sim = SimConfig(
        headless=s.get("headless", False),
        set_joints=s.get("set_joints", True),
        enable_right=s.get("enable_right"),
        enable_left=s.get("enable_left"),
        collision_tracking=s.get("collision_tracking", True),
        desk_prim_paths=s.get("desk_prim_paths", []),
        camera_eye=tuple(s["camera_eye"]) if "camera_eye" in s else (1.97035, 0.00915, 1.58108),
        camera_target=tuple(s["camera_target"]) if "camera_target" in s else (0.51, 0.0, 1.23),
        object_cam=s.get("object_cam", "right"),
        object_trajectory_frame=object_trajectory_frame,
        object_scale=s.get("object_scale", 0.001),
        object_replay=s.get("object_replay", True),
        object_grasp_attach=s.get("object_grasp_attach", True),
        object_grasp_anchor=s.get("object_grasp_anchor", False),
        object_anchor_settle_steps=s.get("object_anchor_settle_steps", 120),
        object_initial_surface_align=s.get("object_initial_surface_align", False),
        object_initial_surface_prim_paths=s.get("object_initial_surface_prim_paths", []),
        object_initial_surface_clearance=s.get("object_initial_surface_clearance", 0.0),
        object_disable_hand_collision=s.get("object_disable_hand_collision", False),
        object_hand_collision_filter_prim_paths=s.get("object_hand_collision_filter_prim_paths", []),
        object_release_collision_delay_frames=s.get("object_release_collision_delay_frames", 0),
        object_free_physics_extra_frames=s.get("object_free_physics_extra_frames", 0),
        object_release_container_align=s.get("object_release_container_align", False),
        object_release_container_prim_path=s.get("object_release_container_prim_path"),
        object_release_container_xy_offset=s.get("object_release_container_xy_offset", [0.0, 0.0]),
        control_hz=s.get("control_hz", 30),
        objects=objects,
    )

    v = raw.get("vis", {})
    vis = VisConfig(
        enabled=v.get("enabled", False),
        show_eef=v.get("show_eef", True),
        show_offset=v.get("show_offset", True),
        video_mode=v.get("video_mode", False),
        eef_alpha=v.get("eef_alpha"),
    )

    sg = raw.get("seg", {})
    seg = SegConfig(
        enabled=sg.get("enabled", False),
        output_dir=sg.get("output_dir", "data/segmentation"),
        save_rate=sg.get("save_rate", 1),
        mask_png=sg.get("mask_png", False),
        mask_npy=sg.get("mask_npy", False),
        rgb_png=sg.get("rgb_png", False),
    )

    lg = raw.get("log", {})
    log = LogConfig(
        file_enabled=lg.get("file_enabled", False),
        file_level=lg.get("file_level", "DEBUG"),
        console_level=lg.get("console_level", "INFO"),
    )

    rec = raw.get("rec", {})
    obj_traj = raw.get("object_trajectory", {})
    rec = RecordingConfig(
        enabled=rec.get("enabled", True),
        output_dir=rec.get("output_dir", "outputs/"),
        cameras=rec.get("cameras", ["viewport"]),
        fps=rec.get("fps", 30),
        start_frame=rec.get("start_frame", 0),
        end_frame=rec.get("end_frame"),
        resolution=tuple(rec.get("resolution", (640, 480))),
        rgb_mp4=rec.get("rgb_mp4", False),
        depth_npy=rec.get("depth_npy", False),
        depth_png=rec.get("depth_png", False),
        eef_pos_npy=rec.get("eef_pos_npy", False),
        eef_pos_mp4=rec.get("eef_pos_mp4", False),
        object_trajectory_npy=rec.get("object_trajectory_npy", False),
        object_trajectory=ObjectTrajectoryRecordingConfig(
            enabled=obj_traj.get("enabled", rec.get("object_trajectory_npy", False)),
            output_dir=obj_traj.get("output_dir"),
            start_frame=obj_traj.get("start_frame", rec.get("start_frame", 0)),
            end_frame=obj_traj["end_frame"] if "end_frame" in obj_traj else rec.get("end_frame"),
        ),
    )

    return scene_path, data_path, robot, sim, vis, seg, log, rec
