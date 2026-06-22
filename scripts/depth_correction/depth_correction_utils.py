"""Shared utilities for depth_correction_fit.py and depth_correction_apply.py."""

import re
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PIL import Image
from scipy.interpolate import RBFInterpolator

_DATA_ID_RE = re.compile(r"\d{8}_\d{6}")


def extract_data_id(path: Path) -> str | None:
    m = _DATA_ID_RE.search(str(path))
    return m.group(0) if m else None


def validate_data_ids(paths: list[Path]) -> str:
    ids = {extract_data_id(p) for p in paths}
    ids.discard(None)
    if len(ids) == 0:
        raise ValueError("No data ID (YYYYMMDD_HHMMSS) found in any input path.")
    if len(ids) > 1:
        raise ValueError(f"Mismatched data IDs across input paths: {sorted(ids)}")
    return ids.pop()


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def save_mask_png(mask: np.ndarray, path: Path) -> None:
    """Save a boolean mask as an 8-bit grayscale PNG (True=255, False=0)."""
    Image.fromarray((mask.astype(np.uint8) * 255)).save(path)
    print(f"  Saved: {path}")


def load_depth_png(path: Path, mm_to_m: float) -> np.ndarray:
    """Load a uint16 depth PNG (mm) and return float32 array in metres."""
    return np.array(Image.open(path), dtype=np.float32) / mm_to_m


# ---------------------------------------------------------------------------
# Mask / foreground computation
# ---------------------------------------------------------------------------

def mask_bounding_box(
    mask: np.ndarray,
    safety_margin: int,
    img_shape: tuple[int, int],
) -> tuple[int, int, int, int]:
    """Return (rmin, rmax, cmin, cmax) of the mask bounding box + safety margin."""
    rows = np.any(mask > 0, axis=1)
    cols = np.any(mask > 0, axis=0)
    rmin, rmax = int(np.where(rows)[0][0]),  int(np.where(rows)[0][-1])
    cmin, cmax = int(np.where(cols)[0][0]),  int(np.where(cols)[0][-1])
    rmin = max(0, rmin - safety_margin)
    rmax = min(img_shape[0] - 1, rmax + safety_margin)
    cmin = max(0, cmin - safety_margin)
    cmax = min(img_shape[1] - 1, cmax + safety_margin)
    return rmin, rmax, cmin, cmax


def build_foreground_mask(
    cfg:           SimpleNamespace,
    depth_real_mm: np.ndarray,
    depth_sim_m:   np.ndarray,
    masks:         list[np.ndarray],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (mask_real, mask_sim, mask_fused).

    mask_real  — finite & non-zero pixels in the real depth image
    mask_sim   — finite & non-zero pixels below background_threshold_m in sim
    mask_fused — intersection of both, with each object's bounding box excluded individually
    """
    mask_real = np.isfinite(depth_real_mm) & (depth_real_mm != 0)
    mask_sim  = (
        np.isfinite(depth_sim_m)
        & (depth_sim_m != 0)
        & (depth_sim_m < cfg.background_threshold_m)
    )

    fused = mask_real & mask_sim
    # exclude each object's bounding box independently to avoid merging distant regions
    for mask in masks:
        if mask.any():
            rmin, rmax, cmin, cmax = mask_bounding_box(
                mask, cfg.mask_safety_margin_px, depth_sim_m.shape
            )
            fused[rmin : rmax + 1, cmin : cmax + 1] = False

    # experimental: restrict training to leftmost 3/4 of the image
    if cfg.exclude_right_quarter:
        cutoff = int(depth_sim_m.shape[1] * 0.75)
        fused[:, cutoff:] = False

    return mask_real, mask_sim, fused


# ---------------------------------------------------------------------------
# Linear transform  corrected = b * real
# ---------------------------------------------------------------------------

def fit_linear(real_m: np.ndarray, sim_m: np.ndarray) -> float:
    """Least-squares scalar b such that b * real ≈ sim (no intercept)."""
    return float(np.dot(real_m, sim_m) / np.dot(real_m, real_m))


def apply_linear(depth_real_m: np.ndarray, b: float) -> np.ndarray:
    return (b * depth_real_m).astype(np.float32)


# ---------------------------------------------------------------------------
# RBF interpolation  corrected = f(real)
# ---------------------------------------------------------------------------

def fit_rbf(
    cfg:    SimpleNamespace,
    real_m: np.ndarray,
    sim_m:  np.ndarray,
    rng:    np.random.Generator,
) -> RBFInterpolator:
    """Fit a 1-D RBF interpolator: real depth → sim depth."""
    if len(real_m) > cfg.rbf_max_samples:
        idx = rng.choice(len(real_m), cfg.rbf_max_samples, replace=False)
        real_m, sim_m = real_m[idx], sim_m[idx]

    return RBFInterpolator(
        real_m[:, np.newaxis],
        sim_m,
        kernel=cfg.rbf_kernel,
        smoothing=cfg.rbf_smoothing,
    )


def apply_rbf(depth_real_m: np.ndarray, rbf: RBFInterpolator) -> np.ndarray:
    """Apply RBF in chunks to limit peak memory usage."""
    flat  = depth_real_m.ravel()
    chunk = 50_000
    out   = np.empty_like(flat)
    for start in range(0, len(flat), chunk):
        sl = flat[start : start + chunk]
        out[start : start + chunk] = rbf(sl[:, np.newaxis])
    return out.reshape(depth_real_m.shape).astype(np.float32)
