import tempfile
import unittest
from pathlib import Path

from src.config import load_config


class ConfigLoadTest(unittest.TestCase):
    def test_loads_object_trajectory_config(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            config_path = base / "config.yaml"
            config_path.write_text(
                """
scene_path: scenes/scene.usd
data_path: data/demo.h5
rec:
  enabled: false
  output_dir: outputs
  start_frame: 5
  end_frame: 20
object_trajectory:
  enabled: true
  output_dir: trajectories
  start_frame: 7
  end_frame: 18
""".lstrip(),
                encoding="utf-8",
            )

            *_, rec_cfg = load_config(config_path, base_dir=base)

            self.assertTrue(rec_cfg.object_trajectory.enabled)
            self.assertEqual(rec_cfg.object_trajectory.output_dir, "trajectories")
            self.assertEqual(rec_cfg.object_trajectory.start_frame, 7)
            self.assertEqual(rec_cfg.object_trajectory.end_frame, 18)

    def test_object_trajectory_defaults_to_rec_frame_range(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            config_path = base / "config.yaml"
            config_path.write_text(
                """
scene_path: scenes/scene.usd
data_path: data/demo.h5
rec:
  enabled: false
  output_dir: outputs
  start_frame: 3
  end_frame: 12
object_trajectory:
  enabled: true
""".lstrip(),
                encoding="utf-8",
            )

            *_, rec_cfg = load_config(config_path, base_dir=base)

            self.assertTrue(rec_cfg.object_trajectory.enabled)
            self.assertIsNone(rec_cfg.object_trajectory.output_dir)
            self.assertEqual(rec_cfg.object_trajectory.start_frame, 3)
            self.assertEqual(rec_cfg.object_trajectory.end_frame, 12)

    def test_loads_object_free_physics_extra_frames(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            config_path = base / "config.yaml"
            config_path.write_text(
                """
scene_path: scenes/scene.usd
data_path: data/demo.h5
sim:
  object_free_physics_extra_frames: 42
""".lstrip(),
                encoding="utf-8",
            )

            _, _, _, sim_cfg, *_ = load_config(config_path, base_dir=base)

            self.assertEqual(sim_cfg.object_free_physics_extra_frames, 42)

    def test_loads_auto_object_free_physics_extra_frames(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            config_path = base / "config.yaml"
            config_path.write_text(
                """
scene_path: scenes/scene.usd
data_path: data/demo.h5
sim:
  object_free_physics_extra_frames: auto
""".lstrip(),
                encoding="utf-8",
            )

            _, _, _, sim_cfg, *_ = load_config(config_path, base_dir=base)

            self.assertEqual(sim_cfg.object_free_physics_extra_frames, "auto")

    def test_loads_object_release_collision_delay_frames(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            config_path = base / "config.yaml"
            config_path.write_text(
                """
scene_path: scenes/scene.usd
data_path: data/demo.h5
sim:
  object_release_collision_delay_frames: 5
""".lstrip(),
                encoding="utf-8",
            )

            _, _, _, sim_cfg, *_ = load_config(config_path, base_dir=base)

            self.assertEqual(sim_cfg.object_release_collision_delay_frames, 5)

    def test_loads_object_initial_surface_align_config(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            config_path = base / "config.yaml"
            config_path.write_text(
                """
scene_path: scenes/scene.usd
data_path: data/demo.h5
sim:
  object_initial_surface_align: true
  object_initial_surface_prim_paths: ["/World/table"]
  object_initial_surface_clearance: 0.002
""".lstrip(),
                encoding="utf-8",
            )

            _, _, _, sim_cfg, *_ = load_config(config_path, base_dir=base)

            self.assertTrue(sim_cfg.object_initial_surface_align)
            self.assertEqual(sim_cfg.object_initial_surface_prim_paths, ["/World/table"])
            self.assertEqual(sim_cfg.object_initial_surface_clearance, 0.002)

    def test_loads_object_hand_collision_filter_config(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            config_path = base / "config.yaml"
            config_path.write_text(
                """
scene_path: scenes/scene.usd
data_path: data/demo.h5
sim:
  object_disable_hand_collision: true
  object_hand_collision_filter_prim_paths:
    - /World/fer_orcahand_right_extended/right_palm
""".lstrip(),
                encoding="utf-8",
            )

            _, _, _, sim_cfg, *_ = load_config(config_path, base_dir=base)

            self.assertTrue(sim_cfg.object_disable_hand_collision)
            self.assertEqual(
                sim_cfg.object_hand_collision_filter_prim_paths,
                ["/World/fer_orcahand_right_extended/right_palm"],
            )

    def test_expands_manipulation_id_in_paths(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            config_path = base / "config.yaml"
            config_path.write_text(
                """
scene_path: scenes/scene.usd
manipulation_id: demo_001
data_path: data/{manipulation_id}/{manipulation_id}.h5
sim:
  objects:
    - usd_path: assets/objects/duck.obj
      trajectory_npy: data/{manipulation_id}/object_initial_pose.npy
      object_state_npy: data/{manipulation_id}/object_states.csv
""".lstrip(),
                encoding="utf-8",
            )

            scene_path, data_path, _, sim_cfg, *_ = load_config(config_path, base_dir=base)

            self.assertEqual(base / "scenes/scene.usd", scene_path)
            self.assertEqual(base / "data/demo_001/demo_001.h5", data_path)
            self.assertEqual(base / "data/demo_001/object_initial_pose.npy", sim_cfg.objects[0].trajectory_npy)
            self.assertEqual(base / "data/demo_001/object_states.csv", sim_cfg.objects[0].object_state)

    def test_loads_world_object_trajectory_frame(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            config_path = base / "config.yaml"
            config_path.write_text(
                """
scene_path: scenes/scene.usd
data_path: data/demo.h5
sim:
  object_trajectory_frame: world
""".lstrip(),
                encoding="utf-8",
            )

            _, _, _, sim_cfg, *_ = load_config(config_path, base_dir=base)

            self.assertEqual("world", sim_cfg.object_trajectory_frame)

    def test_rejects_invalid_object_trajectory_frame(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            config_path = base / "config.yaml"
            config_path.write_text(
                """
scene_path: scenes/scene.usd
data_path: data/demo.h5
sim:
  object_trajectory_frame: robot
""".lstrip(),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "object_trajectory_frame"):
                load_config(config_path, base_dir=base)

    def test_expands_h5_id_from_demo_manipulation_id(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            config_path = base / "config.yaml"
            config_path.write_text(
                """
scene_path: scenes/scene.usd
data_path: data/{manipulation_id}/{h5_id}.h5
""".lstrip(),
                encoding="utf-8",
            )

            _, data_path, *_ = load_config(
                config_path,
                base_dir=base,
                manipulation_id="20250804_105512_demo_0_468",
            )

            self.assertEqual(base / "data/20250804_105512_demo_0_468/20250804_105512.h5", data_path)

    def test_expands_manipulation_id_in_output_dirs(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            config_path = base / "config.yaml"
            config_path.write_text(
                """
scene_path: scenes/scene.usd
manipulation_id: demo_001
data_path: data/{manipulation_id}/{manipulation_id}.h5
seg:
  output_dir: segmentation/{manipulation_id}
rec:
  output_dir: outputs/{manipulation_id}
object_trajectory:
  enabled: true
  output_dir: data/{manipulation_id}/
""".lstrip(),
                encoding="utf-8",
            )

            *_, seg_cfg, _, rec_cfg = load_config(config_path, base_dir=base)

            self.assertEqual(seg_cfg.output_dir, "segmentation/demo_001")
            self.assertEqual(rec_cfg.output_dir, "outputs/demo_001")
            self.assertEqual(rec_cfg.object_trajectory.output_dir, "data/demo_001/")


if __name__ == "__main__":
    unittest.main()
