#!/usr/bin/env python3
"""
Depth correction — fit stage.

Loads one reference real + sim depth frame, builds a foreground mask, fits
linear and/or RBF transforms, evaluates them, and saves:

  outputs/<DATA_ID>/depth_correction/
    transform_params.json        — scale factor b and/or RBF metadata + metrics
    rbf_training_data.npz        — (real, sim) pairs to refit the RBF later
    mask_real.png                — valid-pixel mask for real depth   (optional)
    mask_sim.png                 — valid-pixel mask for sim depth    (optional)
    mask_fused.png               — combined training mask            (optional)
    depth_corrected_<method>.png — sim | corrected | residual        (per method)
    depth_before_after.png       — full-image overview across methods

Config: config/depth_correction_config.yaml  (section: depth_correction)
Run:    python scripts/depth_correction/depth_correction_fit.py
"""

import json
from pathlib import Path
from types import SimpleNamespace

import matplotlib.pyplot as plt
import numpy as np
import yaml
from PIL import Image

from depth_correction_utils import (
    apply_linear,
    apply_rbf,
    build_foreground_mask,
    fit_linear,
    fit_rbf,
    load_depth_png,
    save_mask_png,
    validate_data_ids,
)

ROOT        = Path(__file__).parent.parent.parent
CONFIG_FILE = ROOT / "config" / "depth_correction_config.yaml"


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

def load_config(path: Path = CONFIG_FILE) -> SimpleNamespace:
    with open(path) as fh:
        raw = yaml.safe_load(fh)["depth_correction"]

    all_paths = (
        [ROOT / raw["depth_real_file"], ROOT / raw["depth_sim_file"]]
        + [ROOT / p for p in raw["mask_files"]]
    )
    data_id              = validate_data_ids(all_paths)
    exclude_right_quarter = bool(raw["exclude_right_quarter"])
    dir_name             = "depth_correction_excl_right_quarter" if exclude_right_quarter else "depth_correction"
    output_dir           = ROOT / raw["output_dir"] / data_id / dir_name

    return SimpleNamespace(
        depth_real_file            = ROOT / raw["depth_real_file"],
        depth_sim_file             = ROOT / raw["depth_sim_file"],
        mask_files                 = [ROOT / p for p in raw["mask_files"]],
        output_dir                 = output_dir,
        run_linear                 = bool(raw["run_linear"]),
        run_rbf                    = bool(raw["run_rbf"]),
        mm_to_m                    = float(raw["mm_to_m"]),
        background_threshold_m     = float(raw["background_threshold_m"]),
        mask_safety_margin_px      = int(raw["mask_safety_margin_px"]),
        exclude_right_quarter      = bool(raw["exclude_right_quarter"]),
        rbf_max_samples            = int(raw["rbf_max_samples"]),
        rbf_kernel                 = str(raw["rbf_kernel"]),
        rbf_smoothing              = float(raw["rbf_smoothing"]),
        cmap_depth                 = str(raw["cmap_depth"]),
        cmap_residual              = str(raw["cmap_residual"]),
        residual_vmax_m            = float(raw["residual_vmax_m"]),
        save_foreground_mask_real  = bool(raw["save_foreground_mask_real"]),
        save_foreground_mask_sim   = bool(raw["save_foreground_mask_sim"]),
        save_foreground_mask_fused = bool(raw["save_foreground_mask_fused"]),
    )


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_data(cfg: SimpleNamespace) -> tuple[np.ndarray, np.ndarray, list[np.ndarray]]:
    depth_real_mm = np.array(Image.open(cfg.depth_real_file), dtype=np.float32)
    depth_sim     = load_depth_png(cfg.depth_sim_file, cfg.mm_to_m)
    masks         = [np.array(Image.open(p).convert("L"), dtype=np.uint8) for p in cfg.mask_files]
    return depth_real_mm, depth_sim, masks


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate(pred: np.ndarray, target: np.ndarray, mask: np.ndarray) -> dict:
    diff = pred[mask] - target[mask]
    return {
        "mae_m":  float(np.mean(np.abs(diff))),
        "rmse_m": float(np.sqrt(np.mean(diff ** 2))),
    }


# ---------------------------------------------------------------------------
# Visualisation
# ---------------------------------------------------------------------------

def save_fit_visualization(
    cfg:             SimpleNamespace,
    depth_corrected: np.ndarray,
    depth_sim_m:     np.ndarray,
    depth_real_m:    np.ndarray,
    fg_mask:         np.ndarray,
    method_name:     str,
    metrics:         dict,
) -> None:
    """Save a 3-panel figure: sim | corrected | residual."""
    nan   = np.full(depth_sim_m.shape, np.nan, dtype=np.float32)
    d_sim = np.where(fg_mask, depth_sim_m,     nan)
    d_cor = np.where(fg_mask, depth_corrected, nan)
    d_res = np.where(fg_mask, depth_corrected - depth_sim_m, nan)

    vmin = float(np.nanmin(d_sim))
    vmax = float(np.nanmax(d_sim))

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(
        f"Depth correction — {method_name}  "
        f"[MAE={metrics['mae_m']*100:.1f} cm  RMSE={metrics['rmse_m']*100:.1f} cm]",
        fontsize=13,
    )

    im0 = axes[0].imshow(d_sim, cmap=cfg.cmap_depth, vmin=vmin, vmax=vmax)
    axes[0].set_title("Depth sim (target)")
    plt.colorbar(im0, ax=axes[0], label="m", fraction=0.046, pad=0.04)

    im1 = axes[1].imshow(d_cor, cmap=cfg.cmap_depth, vmin=vmin, vmax=vmax)
    axes[1].set_title(f"Corrected real ({method_name})")
    plt.colorbar(im1, ax=axes[1], label="m", fraction=0.046, pad=0.04)

    im2 = axes[2].imshow(
        d_res, cmap=cfg.cmap_residual,
        vmin=-cfg.residual_vmax_m, vmax=cfg.residual_vmax_m,
    )
    axes[2].set_title("Residual (corrected − sim)")
    plt.colorbar(im2, ax=axes[2], label="m", fraction=0.046, pad=0.04)

    for ax in axes:
        ax.axis("off")

    plt.tight_layout()
    out_path = cfg.output_dir / f"depth_corrected_{method_name}.png"
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


def save_before_after_visualization(
    cfg:              SimpleNamespace,
    depth_real_m:     np.ndarray,
    depth_sim_m:      np.ndarray,
    corrected_linear: np.ndarray | None,
    corrected_rbf:    np.ndarray | None,
) -> None:
    """Save full (unmasked) rows: real | sim | linear corrected | RBF corrected."""
    rows = [
        ("Real depth",   depth_real_m),
        ("Sim depth",    depth_sim_m),
    ]
    if corrected_linear is not None:
        rows.append(("Corrected — linear", corrected_linear))
    if corrected_rbf is not None:
        rows.append(("Corrected — RBF",    corrected_rbf))

    def _clean(title: str, data: np.ndarray) -> np.ndarray:
        invalid = data == 0
        if title == "Sim depth":
            invalid |= data >= cfg.background_threshold_m
        return np.where(invalid, np.nan, data)

    rows = [(title, _clean(title, data)) for title, data in rows]

    all_valid = np.concatenate([d[np.isfinite(d)].ravel() for _, d in rows])
    vmin = float(np.nanpercentile(all_valid, 1))
    vmax = float(np.nanpercentile(all_valid, 99))

    cmap = plt.get_cmap(cfg.cmap_depth).copy()
    cmap.set_bad(color="white")

    fig, axes = plt.subplots(len(rows), 1, figsize=(10, 4 * len(rows)))
    if len(rows) == 1:
        axes = [axes]
    fig.suptitle("Depth overview", fontsize=13)

    for ax, (title, data) in zip(axes, rows):
        im = ax.imshow(data, cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_title(title)
        plt.colorbar(im, ax=ax, label="m", fraction=0.046, pad=0.04)
        ax.axis("off")

    plt.tight_layout()
    out_path = cfg.output_dir / "depth_before_after.png"
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    cfg = load_config()
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(42)

    print("Loading data ...")
    depth_real_mm, depth_sim_m, masks = load_data(cfg)
    depth_real_m = depth_real_mm / cfg.mm_to_m

    mask_real, mask_sim, fg_mask = build_foreground_mask(
        cfg, depth_real_mm, depth_sim_m, masks
    )

    if cfg.save_foreground_mask_real:
        save_mask_png(mask_real, cfg.output_dir / "mask_real.png")
    if cfg.save_foreground_mask_sim:
        save_mask_png(mask_sim,  cfg.output_dir / "mask_sim.png")
    if cfg.save_foreground_mask_fused:
        save_mask_png(fg_mask,   cfg.output_dir / "mask_fused.png")

    real_fg = depth_real_m[fg_mask]
    sim_fg  = depth_sim_m[fg_mask]

    print(f"Foreground pixels: {fg_mask.sum():,} / {fg_mask.size:,}")
    print(f"  real depth range : {real_fg.min():.3f} – {real_fg.max():.3f} m")
    print(f"  sim  depth range : {sim_fg.min():.3f} – {sim_fg.max():.3f} m")

    corrected_linear = None
    metrics_linear   = None
    if cfg.run_linear:
        print("\n[Linear] Fitting ...")
        b = fit_linear(real_fg, sim_fg)
        corrected_linear = apply_linear(depth_real_m, b)
        metrics_linear   = evaluate(corrected_linear, depth_sim_m, fg_mask)
        print(f"  b = {b:.6f}")
        print(f"  MAE  = {metrics_linear['mae_m']*100:.2f} cm")
        print(f"  RMSE = {metrics_linear['rmse_m']*100:.2f} cm")

    corrected_rbf = None
    metrics_rbf   = None
    if cfg.run_rbf:
        print(f"\n[RBF] Fitting (up to {cfg.rbf_max_samples:,} samples, kernel={cfg.rbf_kernel}) ...")
        rbf = fit_rbf(cfg, real_fg, sim_fg, rng)
        print("  Applying RBF to full image ...")
        corrected_rbf = apply_rbf(depth_real_m, rbf)
        metrics_rbf   = evaluate(corrected_rbf, depth_sim_m, fg_mask)
        print(f"  MAE  = {metrics_rbf['mae_m']*100:.2f} cm")
        print(f"  RMSE = {metrics_rbf['rmse_m']*100:.2f} cm")

    if cfg.run_linear and cfg.run_rbf:
        better = "rbf" if metrics_rbf["rmse_m"] < metrics_linear["rmse_m"] else "linear"
        print(f"\nBetter method (lower RMSE): {better.upper()}")
    else:
        better = "rbf" if cfg.run_rbf else "linear"

    transform_data = {
        "description": (
            "Depth correction transforms mapping corrupted real depth (mm) "
            "to corrected depth (m) approximating sim depth."
        ),
        "real_scale_mm_to_m":     cfg.mm_to_m,
        "background_threshold_m": cfg.background_threshold_m,
        "mask_safety_margin_px":  cfg.mask_safety_margin_px,
        "better_method":          better,
    }
    if cfg.run_linear:
        transform_data["linear"] = {
            "formula": "corrected_m = b * (depth_real_mm / mm_to_m)",
            "b":       b,
            "metrics": metrics_linear,
        }
    if cfg.run_rbf:
        transform_data["rbf"] = {
            "kernel":               cfg.rbf_kernel,
            "smoothing":            cfg.rbf_smoothing,
            "max_training_samples": cfg.rbf_max_samples,
            "training_data_file":   "rbf_training_data.npz",
            "metrics":              metrics_rbf,
        }

    params_path = cfg.output_dir / "transform_params.json"
    with open(params_path, "w") as fh:
        json.dump(transform_data, fh, indent=2)
    print(f"\nSaved: {params_path}")

    if cfg.run_rbf:
        rbf_data_path = cfg.output_dir / "rbf_training_data.npz"
        np.savez(rbf_data_path, real_m=real_fg, sim_m=sim_fg)
        print(f"Saved: {rbf_data_path}")

    print("\nGenerating visualisations ...")
    if cfg.run_linear:
        save_fit_visualization(cfg, corrected_linear, depth_sim_m, depth_real_m, fg_mask, "linear", metrics_linear)
    if cfg.run_rbf:
        save_fit_visualization(cfg, corrected_rbf,    depth_sim_m, depth_real_m, fg_mask, "rbf",    metrics_rbf)
    save_before_after_visualization(cfg, depth_real_m, depth_sim_m, corrected_linear, corrected_rbf)

    print("\nDone.")


if __name__ == "__main__":
    main()
