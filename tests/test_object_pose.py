import tempfile
import unittest
from pathlib import Path

import numpy as np

from src.object_pose import load_object_pose_sequence


class ObjectPoseSequenceTest(unittest.TestCase):
    def _save_and_load(self, array, replay_frames=7):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "pose.npy"
            np.save(path, array)
            return load_object_pose_sequence(path, replay_frames)

    def test_static_4x4_broadcasts_to_replay_length(self):
        pose = np.arange(16, dtype=np.float32).reshape(4, 4)

        sequence = self._save_and_load(pose, replay_frames=5)

        self.assertTrue(sequence.is_static)
        self.assertEqual(len(sequence), 5)
        np.testing.assert_array_equal(sequence[0], pose)
        np.testing.assert_array_equal(sequence[4], pose)
        self.assertEqual(sequence.poses.shape, (5, 4, 4))
        self.assertFalse(sequence.poses.flags["OWNDATA"])

    def test_single_frame_trajectory_behaves_as_static(self):
        pose = np.eye(4, dtype=np.float32)
        raw = pose.reshape(1, 4, 4)

        sequence = self._save_and_load(raw, replay_frames=3)

        self.assertTrue(sequence.is_static)
        self.assertEqual(len(sequence), 3)
        np.testing.assert_array_equal(sequence[2], pose)
        self.assertEqual(sequence.source_shape, (1, 4, 4))
        self.assertFalse(sequence.poses.flags["OWNDATA"])

    def test_multi_frame_trajectory_is_preserved(self):
        trajectory = np.stack(
            [
                np.eye(4, dtype=np.float32),
                np.eye(4, dtype=np.float32) * 2,
                np.eye(4, dtype=np.float32) * 3,
            ],
            axis=0,
        )

        sequence = self._save_and_load(trajectory, replay_frames=9)

        self.assertFalse(sequence.is_static)
        self.assertEqual(len(sequence), 3)
        np.testing.assert_array_equal(sequence[1], trajectory[1])
        self.assertEqual(sequence.poses.shape, (3, 4, 4))

    def test_invalid_shape_raises(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "bad.npy"
            np.save(path, np.zeros((4, 3), dtype=np.float32))

            with self.assertRaises(ValueError):
                load_object_pose_sequence(path, replay_frames=5)


if __name__ == "__main__":
    unittest.main()
