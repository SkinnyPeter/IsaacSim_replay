# Real2Sim

## Introduction

Real2Sim replays demonstrations recorded on real Franka robots with Orca hands inside NVIDIA Isaac Sim. It loads mono- or dual-arm motion from HDF5, solves the simulated arm motion with Lula IK, replays per-object pose trajectories, and can export RGB, depth, segmentation, end-effector projections, collisions, and final world-space object trajectories. The repository also includes tools for inspecting recordings, constructing the USD scene, correcting depth, classifying object state, and moving datasets to or from Hugging Face Hub.

## Setup

Install NVIDIA Isaac Sim first, clone this repository, and install `requirements.txt` into the Python environment shipped with that Isaac Sim installation. Copy `.env.example` to `.env` only if you use the Hugging Face utilities, place your private recordings under the ignored `data/` directory, and update `config/config.yaml` with your scene, H5, object asset, and trajectory paths. The simulation entry points must run with Isaac Sim's Python rather than an unrelated system Python.

Make sure you have Isaac Sim set up first: https://docs.isaacsim.omniverse.nvidia.com/5.1.0/installation/download.html

```bash
git clone https://github.com/SkinnyPeter/IsaacSim_replay.git
cd Real2Sim
<isaac-python> -m pip install -r requirements.txt
```

Typical Isaac Sim launchers are `./python.sh` on Linux and `python.bat` on Windows. In the commands below, replace `<isaac-python>` with the launcher or Python executable for your installation.

## Quick start

The checked-in configuration is an example tied to a local demonstration. Run it with:
```bash
<isaac-python> .\python.bat <path to repo>/main.py
```

You can point it at your own data before running.

```yaml
scene_path: scenes/scene.usd
manipulation_id: "<manipulation-id>"
data_path: data/{manipulation_id}/<recording>.h5

sim:
  object_trajectory_frame: world
  objects:
    - usd_path: assets/objects/<object>.obj
      trajectory_npy: data/{manipulation_id}/object_trajectory/<object>.npy
```

Build or refresh the scene if needed, then start the replay:

```bash
<isaac-python> scripts/create_scene.py
<isaac-python> main.py
```

In GUI mode, the **Replay Controls** panel provides pause/play, reset, and frame seeking. The player pauses at the end so the final state remains available for inspection.

## Input layout

Large recordings are deliberately excluded from Git. A direct world-trajectory demonstration normally looks like this:

```text
data/<manipulation-id>/
├── <recording>.h5
├── about.yaml                              # optional frame_start/frame_end metadata
└── object_trajectory/
    ├── <object-a>.npy                      # shape (N, 4, 4)
    └── <object-b>.npy
```

Each trajectory row is a homogeneous transform. Use `sim.object_trajectory_frame: world` when the matrices are already world poses; they are then replayed verbatim. The default `camera` mode is retained for older inputs and applies the selected `T_cam_*_world` transform. Static `(4, 4)` and `(1, 4, 4)` pose files remain supported.

The optional `main_world_traj.py <manipulation-id>` entry point uses `config/world_traj_config.yaml` and additionally aligns robot replay with `frame_start`/`frame_end` from a neighboring `about.yaml`. The normal `main.py` path is sufficient when object row 0 corresponds to robot frame 0.

## Features

- Mono- and dual-arm H5 replay with automatic arm detection.
- Lula IK for FER/Orca robots, including quaternion conversion and a configurable flange offset.
- Direct world-space or legacy camera-space object trajectories.
- Optional state-driven grasp attachment, pre-grasp anchoring, collision filtering, and physics release for legacy reconstructed trajectories.
- GUI playback controls plus headless execution.
- RGB MP4, depth NumPy/PNG, EEF pixel positions, and annotated EEF video recording.
- Robot segmentation through Isaac Sim Replicator.
- Final simulated object trajectory recording.
- H5 inspection, camera preview, depth correction, and Hugging Face dataset utilities.

## Repository layout

```text
Real2Sim/
├── main.py                         # standard simulation entry point
├── main_world_traj.py              # world trajectory + metadata frame-window entry point
├── config/                         # runtime and utility YAML files
├── src/                            # loaders, configuration, and simulator implementation
├── scripts/                        # scene, H5, depth, state, and dataset utilities
├── watch-demo/                     # lightweight H5 inspection and playback
├── assets/                         # robot and example object assets
├── scenes/                         # USD scene files
├── tests/                          # non-rendering unit tests
└── data/                           # local recordings; ignored by Git
```

## Configuration

See [config/README.md](config/README.md) for the complete field reference. The main sections are:

| Section | Purpose |
|---|---|
| `robot` | Robot prims, EEF frames, flange offset, and calibrated camera-to-world transforms |
| `sim` | Replay rate, arm selection, object pose frame, physics, collisions, and grasp behavior |
| `sim.objects[]` | Object asset, pose trajectory, optional state data, prim path, and grasp settings |
| `vis` | EEF debug visualization |
| `seg` | Segmentation and RGB export |
| `rec` | RGB, depth, and EEF recording |
| `object_trajectory` | Final simulated object world-pose recording |
| `log` | Console and optional file logging |

For direct world trajectories, disable reconstruction features such as `object_grasp_attach`, `object_grasp_anchor`, `object_initial_surface_align`, and `object_release_container_align`. Also avoid recording object trajectories back into the same directory as the inputs.

## Outputs

Depending on the enabled options, a recording directory can contain:

```text
outputs/<recording-id>/
├── mp4/<recording-id>_<camera>.mp4
├── depth/<camera>/npy/000000.npy
├── depth/<camera>/png/000000.png
├── EEF_pos/*.npy
├── EEF_pos/*.mp4
└── object_trajectory/<object-name>.npy
```

Object trajectory recording follows natural forward playback and ignores pause, seek, and reset interactions.

## Utilities

| Command | Purpose |
|---|---|
| `<isaac-python> scripts/create_scene.py` | Create or refresh `scenes/scene.usd` |
| `python watch-demo/inspect_h5.py <file.h5>` | Print H5 structure or plot trajectories |
| `python watch-demo/play_h5_video.py <file.h5>` | Interactively preview H5 camera streams |
| `python scripts/h5_analyzer.py` | Inspect known robot datasets and quaternion conventions |
| `python scripts/view_h5_video.py <file.h5>` | Preview or export an H5 video stream |
| `python scripts/detect_object_state.py <id>` | Classify static, moving, and grasped frames |
| `python scripts/depth_correction/depth_correction_fit.py` | Fit a real-to-sim depth correction |
| `python scripts/depth_correction/depth_correction_apply.py` | Apply a fitted depth correction |

Hugging Face commands and their differing supported layouts are documented in [scripts/huggingface/README.md](scripts/huggingface/README.md) and [hugging-face/README.md](hugging-face/README.md).

## Tests

The unit tests cover configuration expansion, pose sequence loading, object trajectory recording, and world-trajectory frame metadata. They do not launch Isaac Sim.

```bash
python -m unittest discover -s tests -v
```

Run the simulation itself as an integration check after changing Isaac Sim APIs, scene assets, or robot descriptions.

## Public-release notes

- `data/`, `outputs/`, `.env`, logs, caches, and generated media are ignored. Never commit access tokens or private recordings.
- The example YAML files contain calibration values and demonstration-shaped paths; replace them for your setup.
- The bundled robot description contains third-party license and notice files under `assets/pandaorca_description-main/licenses/`.
- Before redistributing the repository, add a root project license and verify redistribution rights for every object mesh, robot mesh, and USD asset. A public GitHub repository without a root license does not grant reuse rights.
- The repository contains large binary/mesh assets. The largest tracked file is below GitHub's single-file limit, but Git LFS or downloadable release assets may make cloning substantially lighter.
