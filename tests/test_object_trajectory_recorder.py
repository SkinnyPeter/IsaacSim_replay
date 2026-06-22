import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from src.config import ObjectTrajectoryRecordingConfig, RecordingConfig
from src.simulator.object_trajectory_recorder import ObjectTrajectoryRecorder, PROJECT_ROOT


class _FakePrim:
    def __init__(self):
        self.frame = 0

    def get_world_poses(self):
        position = np.array([[float(self.frame), float(self.frame + 1), float(self.frame + 2)]])
        orientation_wxyz = np.array([[1.0, 0.0, 0.0, 0.0]])
        return position, orientation_wxyz


class _FakeObject:
    def __init__(self, name="cube", prim_path="/World/cube"):
        self.name = name
        self.prim_path = prim_path
        self.prim = _FakePrim()


class ObjectTrajectoryRecorderTest(unittest.TestCase):
    def test_relative_output_dir_resolves_under_project_root(self):
        recorder = ObjectTrajectoryRecorder(RecordingConfig(output_dir="outputs"))

        output_root = recorder._resolve_output_root(Path(r"C:\isaacsim\data\demo.h5"))

        self.assertEqual(output_root, PROJECT_ROOT / "outputs")

    def test_absolute_output_dir_is_preserved(self):
        output_dir = Path(r"C:\tmp\trajectories")
        recorder = ObjectTrajectoryRecorder(RecordingConfig(output_dir=str(output_dir)))

        self.assertEqual(recorder._resolve_output_root(None), output_dir)

    def test_records_only_configured_object_trajectory_frame_range(self):
        with TemporaryDirectory() as temp_dir:
            obj = _FakeObject()
            recorder = ObjectTrajectoryRecorder(
                RecordingConfig(
                    output_dir=temp_dir,
                    object_trajectory=ObjectTrajectoryRecordingConfig(
                        enabled=True,
                        output_dir=temp_dir,
                        start_frame=2,
                        end_frame=4,
                    ),
                )
            )

            recorder.start([obj], h5_path=Path("demo.h5"), n_frames=10)
            for frame in range(7):
                obj.prim.frame = frame
                recorder.capture(frame, [obj])
            recorder.stop()

            trajectory = np.load(Path(temp_dir) / "demo" / "object_trajectory" / "cube.npy")

        self.assertEqual(trajectory.shape, (3, 4, 4))
        np.testing.assert_allclose(trajectory[:, :3, 3], [[2.0, 3.0, 4.0], [3.0, 4.0, 5.0], [4.0, 5.0, 6.0]])

    def test_null_object_trajectory_end_records_until_replay_end(self):
        with TemporaryDirectory() as temp_dir:
            obj = _FakeObject()
            recorder = ObjectTrajectoryRecorder(
                RecordingConfig(
                    output_dir=temp_dir,
                    object_trajectory=ObjectTrajectoryRecordingConfig(
                        enabled=True,
                        output_dir=temp_dir,
                        start_frame=3,
                        end_frame=None,
                    ),
                )
            )

            recorder.start([obj], h5_path=Path("demo.h5"), n_frames=5)
            for frame in range(5):
                obj.prim.frame = frame
                recorder.capture(frame, [obj])
            recorder.stop()

            trajectory = np.load(Path(temp_dir) / "demo" / "object_trajectory" / "cube.npy")

        self.assertEqual(trajectory.shape, (2, 4, 4))
        np.testing.assert_allclose(trajectory[:, :3, 3], [[3.0, 4.0, 5.0], [4.0, 5.0, 6.0]])


if __name__ == "__main__":
    unittest.main()
