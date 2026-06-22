import logging
import numpy as np
from pathlib import Path

from src.config import RobotConfig, SegConfig

logger = logging.getLogger(__name__)


class Segmentor:
    def __init__(self, stage, robot_config: RobotConfig, seg_config: SegConfig):
        self.stage = stage
        self.robot_config = robot_config
        self.seg_config = seg_config
        self._seg_annotators: dict = {}
        self._rgb_annotators: dict = {}
        self._seg_dirs: dict = {}

    @property
    def active(self) -> bool:
        return self.seg_config.enabled

    def setup(self):
        if not self.seg_config.enabled:
            return

        need_mask = self.seg_config.mask_png or self.seg_config.mask_npy
        need_rgb = self.seg_config.rgb_png
        logger.info("[seg] setup: mask_png=%s mask_npy=%s rgb_png=%s save_rate=%d output_dir=%s",
                    self.seg_config.mask_png, self.seg_config.mask_npy,
                    self.seg_config.rgb_png, self.seg_config.save_rate, self.seg_config.output_dir)

        if not need_mask and not need_rgb:
            logger.warning("[seg] enabled=True but all output flags are off (mask_png, mask_npy, rgb_png) — nothing will be saved")
            return

        import omni.kit.app
        omni.kit.app.get_app().get_extension_manager().set_extension_enabled_immediate(
            "omni.replicator.core", True
        )
        import omni.replicator.core as rep

        if need_mask:
            self._tag_robot_prims(rep)

        for side, cam_path in (("left", "/World/camera_left"), ("right", "/World/camera_right")):
            if not self.stage.GetPrimAtPath(cam_path).IsValid():
                logger.warning("[seg] camera prim not found: %s — skipping %s", cam_path, side)
                continue
            rp = rep.create.render_product(cam_path, (640, 480))
            if need_mask:
                ann = rep.AnnotatorRegistry.get_annotator(
                    "semantic_segmentation", init_params={"colorize": False}
                )
                ann.attach([rp])
                self._seg_annotators[side] = ann
            if need_rgb:
                ann_rgb = rep.AnnotatorRegistry.get_annotator("rgb")
                ann_rgb.attach([rp])
                self._rgb_annotators[side] = ann_rgb
            logger.info("[seg] attached annotators for %s camera (%s)", side, cam_path)

        if not self._seg_annotators and not self._rgb_annotators:
            logger.warning("[seg] no annotators attached — both cameras missing from stage, capture will be a no-op")
            return

        base = Path(self.seg_config.output_dir)
        for side in ("left", "right"):
            dirs = {}
            if need_rgb:
                dirs["rgb"] = base / side / "rgb"
                dirs["rgb"].mkdir(parents=True, exist_ok=True)
            if self.seg_config.mask_png:
                dirs["seg_png"] = base / side / "seg_png"
                dirs["seg_png"].mkdir(parents=True, exist_ok=True)
            if self.seg_config.mask_npy:
                dirs["seg_npy"] = base / side / "seg_npy"
                dirs["seg_npy"].mkdir(parents=True, exist_ok=True)
            self._seg_dirs[side] = dirs
        logger.info("[seg] output dirs created under %s", base)

    def capture(self, frame: int):
        if frame % self.seg_config.save_rate != 0:
            return

        from PIL import Image

        for side, ann in self._seg_annotators.items():
            data = ann.get_data()
            id_mask = data.get("data")
            if id_mask is None:
                continue

            id_to_labels = data.get("info", {}).get("idToLabels", {})
            robot_ids = {
                int(sid)
                for sid, labels in id_to_labels.items()
                if isinstance(labels, dict) and str(labels.get("class", "")).startswith("robot")
            }
            if not robot_ids:
                logger.warning("[seg] frame %d %s: no 'robot' IDs in idToLabels=%s", frame, side, id_to_labels)
            binary_mask = np.zeros(id_mask.shape, dtype=np.uint8)
            for rid in robot_ids:
                binary_mask[id_mask == rid] = 255

            dirs = self._seg_dirs[side]
            if self.seg_config.mask_npy:
                np.save(dirs["seg_npy"] / f"{frame:06d}.npy", binary_mask)
            if self.seg_config.mask_png:
                Image.fromarray(binary_mask, mode="L").save(dirs["seg_png"] / f"{frame:06d}.png")

        for side, ann in self._rgb_annotators.items():
            rgb = ann.get_data()
            if rgb is None:
                continue
            Image.fromarray(rgb[..., :3]).save(self._seg_dirs[side]["rgb"] / f"{frame:06d}.png")

    def _tag_robot_prims(self, rep):
        """Apply semantic labels via Replicator's legacy API so they appear in idToLabels.

        Skips collision prims — tagging those invalidates the physics tensor view.
        """
        from pxr import Usd

        robot_roots = [self.robot_config.franka_right_path, self.robot_config.franka_left_path]
        tagged = 0
        skipped_collision = 0
        for root_path in robot_roots:
            root_prim = self.stage.GetPrimAtPath(root_path)
            if not root_prim.IsValid():
                logger.warning("[seg] robot prim not found: %s", root_path)
                continue
            for prim in Usd.PrimRange(root_prim):
                path_str = prim.GetPath().pathString
                if "collision" in path_str.lower():
                    skipped_collision += 1
                    continue
                rep.utils._set_semantics_legacy(prim, [("class", "robot")])
                tagged += 1
        logger.debug("[seg] semantic tagging: tagged=%d skipped_collision=%d", tagged, skipped_collision)
