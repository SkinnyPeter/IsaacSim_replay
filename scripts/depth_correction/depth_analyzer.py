import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

INF_CAP_M = 5.0
BIN_WIDTH_M = 0.1
FAR_THRESHOLD_M = 1.2


def _first_npy(depth_npy_dir: Path) -> Path:
    files = sorted(depth_npy_dir.glob("*.npy"))
    if not files:
        raise FileNotFoundError(f"No .npy files found in {depth_npy_dir}")
    return files[0]


def load_depth_dir(depth_npy_dir: Path) -> np.ndarray:
    """Load the first .npy depth frame from a directory as a flat float32 array."""
    f = _first_npy(depth_npy_dir)
    print(f"Using frame: {f.name}")
    return np.load(f).astype(np.float32).ravel()


def depth_value_histogram(depth_npy_dir: str):
    """
    Map depth values to how often they occur across all frames in a directory.
    Inf/NaN are capped to INF_CAP_M before binning.
    """
    values = load_depth_dir(Path(depth_npy_dir))

    n_total = values.size
    n_inf = np.sum(~np.isfinite(values))
    n_neg = np.sum((values < 0) & np.isfinite(values))

    # Replace nan/inf then clip everything to [0, INF_CAP_M]
    values = np.nan_to_num(values, nan=INF_CAP_M, posinf=INF_CAP_M, neginf=0.0)
    values = np.clip(values, 0.0, INF_CAP_M)

    bins = np.arange(0.0, INF_CAP_M + BIN_WIDTH_M, BIN_WIDTH_M)
    counts, edges = np.histogram(values, bins=bins)

    print(f"\n{'Depth range (m)':<22} {'Count':>12} {'%':>8}")
    print("-" * 44)
    for i, count in enumerate(counts):
        lo, hi = edges[i], edges[i + 1]
        label = f"{lo:.1f} – {hi:.1f}"
        if hi >= INF_CAP_M:
            label += "  (≥inf capped)"
        pct = 100.0 * count / n_total
        print(f"  {label:<20} {count:>12,d} {pct:>7.2f}%")

    print(f"\nTotal pixels : {n_total:,d}")
    print(f"Inf/NaN pixels (capped to {INF_CAP_M} m): {n_inf:,d}  ({100*n_inf/n_total:.2f}%)")
    if n_neg:
        print(f"Negative values (clamped to 0): {n_neg:,d}")

    # Plot
    centers = (edges[:-1] + edges[1:]) / 2
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(centers, counts, width=BIN_WIDTH_M * 0.9, align="center", color="steelblue", edgecolor="white")
    ax.set_xlabel("Depth (m)")
    ax.set_ylabel("Pixel count")
    ax.set_title(f"Depth value distribution  [inf capped at {INF_CAP_M} m]\n{Path(depth_npy_dir).resolve()}")
    ax.xaxis.set_major_locator(plt.MultipleLocator(0.5))
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{int(x):,}"))
    plt.tight_layout()
    out_path = Path(depth_npy_dir) / "depth_histogram.png"
    fig.savefig(out_path, dpi=150)
    print(f"\nHistogram saved → {out_path}")
    plt.show()


def highlight_far_pixels(depth_npy_dir: str, threshold_m: float = FAR_THRESHOLD_M):
    """
    Visualize one depth frame: pixels beyond threshold_m are yellow,
    closer pixels are shown with a grayscale colormap. Saves result as PNG.
    """
    npy_path = _first_npy(Path(depth_npy_dir))
    print(f"Using frame: {npy_path.name}")

    depth = np.load(npy_path).astype(np.float32)
    far_mask = (~np.isfinite(depth)) | (depth > threshold_m)

    # Normalize valid range [0, threshold_m] → [0, 1] for colormap
    valid = np.clip(depth, 0.0, threshold_m) / threshold_m

    cmap = plt.get_cmap("gray")
    rgb = cmap(valid)[..., :3]  # (H, W, 3), float64 in [0, 1]

    # Paint far pixels yellow
    rgb[far_mask] = (1.0, 1.0, 0.0)

    img_uint8 = (rgb * 255).astype(np.uint8)

    import cv2
    out_path = npy_path.parent / f"{npy_path.stem}_far_highlight.png"
    cv2.imwrite(str(out_path), cv2.cvtColor(img_uint8, cv2.COLOR_RGB2BGR))

    n_far = far_mask.sum()
    n_total = far_mask.size
    print(f"Far pixels (> {threshold_m} m): {n_far:,d} / {n_total:,d}  ({100*n_far/n_total:.1f}%)")
    print(f"Saved → {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze depth value distribution across recorded .npy frames.")
    parser.add_argument("depth_npy_dir", help="Path to a *_depth/npy/ directory produced by SimulationRecorder")
    parser.add_argument(
        "--highlight-far",
        action="store_true",
        help=f"Save a PNG with pixels beyond {FAR_THRESHOLD_M} m highlighted in yellow",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=FAR_THRESHOLD_M,
        help=f"Distance threshold in meters for --highlight-far (default: {FAR_THRESHOLD_M})",
    )
    args = parser.parse_args()

    if args.highlight_far:
        highlight_far_pixels(args.depth_npy_dir, threshold_m=args.threshold)
    else:
        depth_value_histogram(args.depth_npy_dir)
