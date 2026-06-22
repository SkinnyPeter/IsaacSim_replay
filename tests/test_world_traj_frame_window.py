import tempfile
import unittest
from pathlib import Path

from src.simulator.world_traj.frame_window import FrameWindow, load_frame_window


class FrameWindowTest(unittest.TestCase):
    def test_missing_metadata_defaults_to_full_replay(self):
        with tempfile.TemporaryDirectory() as tmp:
            window = load_frame_window(Path(tmp) / "about.yaml")

        self.assertEqual(window, FrameWindow())
        self.assertIsNone(window.length)

    def test_loads_inclusive_frame_window(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "about.yaml"
            path.write_text("frame_start: 1129\nframe_end: 1571\n", encoding="utf-8")

            window = load_frame_window(path)

        self.assertEqual(window.start, 1129)
        self.assertEqual(window.end, 1571)
        self.assertEqual(window.length, 443)

    def test_rejects_invalid_window(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "about.yaml"
            path.write_text("frame_start: 10\nframe_end: 9\n", encoding="utf-8")

            with self.assertRaises(ValueError):
                load_frame_window(path)


if __name__ == "__main__":
    unittest.main()
