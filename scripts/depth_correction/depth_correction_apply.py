#!/usr/bin/env python3
"""
Depth correction — apply stage.

Loads a fitted transform from transform_params.json (produced by
depth_correction_fit.py) and applies it to every *.png file in an input
directory, saving corrected uint16 PNGs to the output directory.

Config: config/depth_correction_config.yaml  (section: apply)

Usage:
    python scripts/depth_correction/depth_correction_apply.py
"""

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import yaml
from PIL import Image

from depth_correction_utils import (
    apply_linear,
    apply_rbf,
    fit_rbf,
    load_depth_png,
)

ROOT        = Path(__file__).parent.parent.parent
CONFIG_FILE = ROOT / "config" / "depth_correction_config.yaml"


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

def load_config(path: Path = CONFIG_FILE) -> SimpleNamespace:
    with open(path) as fh:
        raw = yaml.safe_load(fh)["apply"]

    return SimpleNamespace(
        transform_params_file = ROOT / raw["transform_params_file"],
        input_dir             = ROOT / raw["input_dir"],
        output_dir            = ROOT / raw["output_dir"],
        method                = str(raw["method"]),
    )


# ---------------------------------------------------------------------------
# Transform loading
# ---------------------------------------------------------------------------

def load_transform(cfg: SimpleNamespace) -> SimpleNamespace:
    """Load transform_params.json and return a SimpleNamespace ready to apply."""
    with open(cfg.transform_params_file) as fh:
        params = json.load(fh)

    if cfg.method not in params:
        available = [k for k in ("linear", "rbf") if k in params]
        raise ValueError(
            f"Method '{cfg.method}' not found in transform_params.json. "
            f"Available: {available}"
        )

    mm_to_m = float(params["real_scale_mm_to_m"])

    if cfg.method == "linear":
        b = float(params["linear"]["b"])
        return SimpleNamespace(method="linear", mm_to_m=mm_to_m, b=b)

    # RBF: reload training data and refit
    rbf_meta      = params["rbf"]
    training_file = cfg.transform_params_file.parent / rbf_meta["training_data_file"]
    data          = np.load(training_file)
    rng           = np.random.default_rng(42)
    rbf_cfg       = SimpleNamespace(
        rbf_max_samples = int(rbf_meta["max_training_samples"]),
        rbf_kernel      = str(rbf_meta["kernel"]),
        rbf_smoothing   = float(rbf_meta["smoothing"]),
    )
    rbf = fit_rbf(rbf_cfg, data["real_m"], data["sim_m"], rng)
    return SimpleNamespace(method="rbf", mm_to_m=mm_to_m, rbf=rbf)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    cfg = load_config()
    cfg.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading transform ({cfg.method}) from {cfg.transform_params_file} ...")
    transform = load_transform(cfg)

    frames = sorted(cfg.input_dir.glob("*.png"))
    if not frames:
        raise FileNotFoundError(f"No PNG files found in {cfg.input_dir}")

    print(f"Applying to {len(frames)} frame(s) → {cfg.output_dir}")
    for frame_path in frames:
        depth_real_m = load_depth_png(frame_path, transform.mm_to_m)

        if transform.method == "linear":
            corrected = apply_linear(depth_real_m, transform.b)
        else:
            corrected = apply_rbf(depth_real_m, transform.rbf)

        # save as uint16 mm (same format as input)
        corrected_mm = np.clip(corrected * transform.mm_to_m, 0, 65535).astype(np.uint16)
        out_path = cfg.output_dir / frame_path.name
        Image.fromarray(corrected_mm).save(out_path)

    print(f"Done. {len(frames)} frame(s) saved to {cfg.output_dir}")


if __name__ == "__main__":
    main()
