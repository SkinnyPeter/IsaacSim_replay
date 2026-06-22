# Config Field Reference

This directory contains the simulation configuration and auxiliary depth/state-processing configurations. Paths are resolved relative to the repository root when loaded by `main.py`; `{manipulation_id}` and `{h5_id}` placeholders are expanded by `src/config.py`. The checked-in values are examples from one local demonstration and must be changed for another dataset.

## config.yaml

---

### Top-level

| Field | Description |
|---|---|
| `scene_path` | Path to the USD scene file |
| `manipulation_id` | Optional quoted recording/manipulation id available in path templates as `{manipulation_id}` |
| `data_path` | Path to the HDF5 dataset used for replay |

---

### `robot`

| Field | Description |
|---|---|
| `franka_right_path` | USD prim path of the right robot |
| `franka_left_path` | USD prim path of the left robot |
| `ee_frame_name` | End-effector frame name in the URDF (used by LulaKinematicsSolver) |
| `ee_usd_prim_name` | End-effector prim name in the USD scene |
| `ee_flange_to_eef_offset` | `[x, y, z]` offset from flange to end-effector |
| `T_cam_left_world` | 4×4 transform from left-camera coordinates to world coordinates |
| `T_cam_right_world` | 4×4 transform from right-camera coordinates to world coordinates |

---

### `sim`

| Field | Default | Description |
|---|---|---|
| `headless` | `false` | Run without GUI |
| `set_joints` | `true` | Set joint positions directly each frame (vs. controller) |
| `enable_right` | _(auto)_ | Enable right arm; `null` infers from H5 structure |
| `enable_left` | _(auto)_ | Enable left arm; `null` infers from H5 structure |
| `control_hz` | `50` | Control loop frequency in Hz |
| `collision_tracking` | `true` | Count frames where hand contacts the environment |
| `desk_prim_paths` | `[]` | USD prim paths of desk prims to track collisions against (empty = all env prims) |
| `object_replay` | `true` | Enable object trajectory replay each frame |
| `object_grasp_attach` | `true` | Attach grasped objects to the configured hand palm during `GRASPED` frames |
| `object_grasp_anchor` | `false` | Hold pre-grasp objects at a settled anchor and release them to physics after the grasp |
| `object_anchor_settle_steps` | `120` | Non-rendering physics steps used to settle an anchored object |
| `object_initial_surface_align` | `false` | Shift initial object poses in Z so oriented object bounds touch the table surface |
| `object_initial_surface_prim_paths` | `[]` | Surface/table prims used for initial alignment; empty uses `desk_prim_paths` |
| `object_initial_surface_clearance` | `0.0` | Extra meters above the detected surface after alignment |
| `object_disable_hand_collision` | `false` | Filter object collision pairs against hand links while preserving object-table and object-object collisions |
| `object_hand_collision_filter_prim_paths` | `[]` | Explicit hand/filter prim paths; empty auto-detects hand collision prims under robot roots |
| `object_release_collision_delay_frames` | `0` | Optional frames to keep the released object kinematic with collisions off before enabling dynamic physics |
| `object_free_physics_extra_frames` | `0` | Extra frames after object trajectory end where robot replay continues and objects run with physics; may be `auto` |
| `object_release_container_align` | `false` | Move a configured container under the releasing hand before replay |
| `object_release_container_prim_path` | `null` | USD prim path of the container moved by release alignment |
| `object_release_container_xy_offset` | `[0,0]` | Additional world XY offset applied during container alignment |
| `object_cam` | `"right"` | Camera frame that object trajectories are expressed in (`"left"` or `"right"`) |
| `object_trajectory_frame` | `"camera"` | Pose coordinate frame: `"camera"` applies `T_cam_*_world`; `"world"` replays transforms directly |
| `object_scale` | `0.001` | Uniform local scale applied to loaded object assets |
| `camera_eye` | `[1.97, 0.009, 1.58]` | Debug viewport camera position `[x, y, z]` |
| `camera_target` | `[0.51, 0.0, 1.23]` | Debug viewport camera look-at target `[x, y, z]` |
| `objects[].usd_path` | — | Path to the object USD asset |
| `objects[].trajectory_npy` | — | Path to object pose data: static `4×4` / `1×4×4` initial pose or `N×4×4` trajectory |
| `objects[].prim_path` | _(auto)_ | USD prim path; defaults to `/World/<usd_stem>` |
| `objects[].object_state` | _(optional)_ | Path to a state `.npy` or `.csv`; enables per-frame grasped/static/moving logic |
| `objects[].object_state_npy` | _(optional)_ | Backward-compatible alias for `objects[].object_state` |
| `objects[].grasp_eef_side` | _(optional)_ | Which arm grasps this object (`"left"` or `"right"`) |
| `objects[].grasp_eef_offset` | `[0,0,0]` | Local-frame offset (m) from palm prim to object center when grasped |
| `objects[].grasp_eef_ori_flip` | _(optional)_ | Extra rotation applied on top of palm orientation when grasped; format `axis:degrees` (e.g. `x:90`, `y:-90`) or `none` |

For `object_trajectory_frame: world`, the simulator applies each pose directly and bypasses camera conversion and state-driven reconstruction during object stepping. Disable anchoring, surface/container alignment, grasp attachment, and free-physics tails unless you intentionally want additional physics setup around the recorded poses. Do not point enabled object-trajectory recording at the same input directory.

---

### `vis`

| Field | Default | Description |
|---|---|---|
| `enabled` | `false` | Enable the EEF visualizer |
| `show_eef` | `true` | Draw IK-target and actual EEF axis frames in the viewport |
| `show_offset` | `true` | Draw a second set of frames offset in Z for orientation inspection |
| `video_mode` | `false` | Use a lower alpha for the faded frames (suited for video output) |
| `eef_alpha` | _(auto)_ | Override alpha for faded frames; `null` uses `video_mode` default |

---

### `log`

| Field | Default | Description |
|---|---|---|
| `file_enabled` | `false` | Write logs to a timestamped file under `logs/` |
| `file_level` | `"DEBUG"` | Level written to file (`DEBUG`, `INFO`, `WARNING`, …) |
| `console_level` | `"INFO"` | Level written to stdout |

---

### `seg`

| Field | Default | Description |
|---|---|---|
| `enabled` | `false` | Enable the segmentation pipeline |
| `output_dir` | `"data/segmentation"` | Directory where segmentation outputs are saved |
| `save_rate` | `1` | Save outputs every N frames |
| `mask_png` | `false` | Save binary robot mask as PNG |
| `mask_npy` | `false` | Save binary robot mask as NumPy array |
| `rgb_png` | `false` | Save RGB frames as PNG |

---

### `rec`

| Field | Default | Description |
|---|---|---|
| `enabled` | `false` | Enable the recording pipeline |
| `output_dir` | `"outputs/"` | Directory for recording outputs |
| `cameras` | `["viewport"]` | Cameras to record; any of `"viewport"`, `"left"`, `"right"` |
| `fps` | `30` | Output video frame rate |
| `start_frame` | `0` | First simulation frame to record |
| `end_frame` | `null` | Last simulation frame to record; `null` = until end |
| `resolution` | `[640, 480]` | Recording resolution `[width, height]` |
| `rgb_mp4` | `false` | Record RGB video as MP4 |
| `depth_npy` | `false` | Save depth frames as `.npy` files |
| `depth_png` | `false` | Save depth frames as 16-bit PNG (millimetres) |
| `eef_pos_npy` | `false` | Save EEF image-space positions per frame as `.npy` under `EEF_pos/` |
| `eef_pos_mp4` | `false` | Save annotated MP4 with red dot and frame/position overlay under `EEF_pos/` |

---

### `object_trajectory`

Records final simulated object world poses after each simulation step. This is independent of camera recording and writes one `N x 4 x 4` homogeneous transform array per active object to `<output_dir>/<recording_id>/object_trajectory/<object_name>.npy`.

| Field | Default | Description |
|---|---|---|
| `enabled` | `false` | Enable object trajectory recording |
| `output_dir` | `null` | Output root; `null` uses `rec.output_dir` |
| `start_frame` | `rec.start_frame` | First simulation frame to record |
| `end_frame` | `rec.end_frame` | Last simulation frame to record; `null` = until replay end |

---

## object_state_config.yaml

Used by `scripts/detect_object_state.py`. All thresholds operate on pixel coordinates from the Aria RGB camera.

---

### Top-level

| Field | Default | Description |
|---|---|---|
| `id` | — | Recording ID to process; overridden by the CLI positional argument |
| `output_dir` | `"outputs/"` | Root output directory; script writes to `<output_dir>/<id>/object_state/` |
| `data_dir` | `"data/h5/"` | Directory containing `<id>.h5` files |
| `object_name` | — | Subdirectory name under `obj_center/` in the H5 file |
| `smooth_window` | `4` | Gaussian sigma in frames applied to EEF and object-center position arrays |
| `pos_threshold` | `50.0` | Max EEF↔object-center pixel distance to satisfy the position condition for grasp |
| `motion_window` | `60` | Half-window in frames over which average pixel speeds are computed for grasp motion check |
| `motion_threshold` | `0.2` | Max difference in average pixel speed between EEF and object center (px/frame) for grasp |
| `static_threshold` | `0.3` | Max object-center pixel speed (px/frame) to classify a non-grasped frame as static |
| `static_motion_window` | `10` | Half-window in frames for the object-center speed used in the static/moving classification |
| `max_grasp_gap` | `20` | Fill gaps in the grasped state shorter than this many frames before applying `min_grasp_duration` |
| `min_grasp_duration` | `15` | Minimum consecutive frames both grasp conditions must hold before a grasp is confirmed |

---

### `video`

| Field | Default | Description |
|---|---|---|
| `enabled` | `true` | Write an annotated MP4 alongside `object_state.npy` |
| `fps` | `30` | Output video frame rate |
