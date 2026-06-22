import h5py
import numpy as np

H5_DEFAULT_PATH = "data/20250827_151212.h5"


class H5Analyzer:
    """
    Utility class for inspecting and visualizing data stored in an HDF5 file.

    The class provides tools to:
    - Print dataset names, shapes, and data types
    - Play image sequences stored in the file as a video

    Parameters
    ----------
    file_path : str, optional
        Path to the HDF5 file to analyze. Defaults to `H5_DEFAULT_PATH`.
    """

    def __init__(self, file_path=H5_DEFAULT_PATH):
        """
        Initialize the analyzer with the path to an HDF5 file.

        Parameters
        ----------
        file_path : str
            Path to the HDF5 file.
        """
        self.file_path = file_path

    def inspect(self):
        """
        Inspect the HDF5 file structure.

        This method prints:
        - Top-level keys in the file
        - All groups and datasets
        - Dataset shapes and data types

        It also attempts to load and display information for common
        joint position datasets (`qpos_arm` and `qpos_hand`) if they exist.
        """
        with h5py.File(self.file_path, "r") as f:
            print("Top-level keys:", list(f.keys()))

            # Print detailed info for all datasets
            def print_item(name, obj):
                if isinstance(obj, h5py.Group):
                    print(f"GROUP:   {name}")
                elif isinstance(obj, h5py.Dataset):
                    print(f"DATASET: {name}")
                    print(f"  shape: {obj.shape}")
                    print(f"  dtype: {obj.dtype}")

            f.visititems(print_item)

            print("\n===== COMMON DATASETS =====")

            # List of keys that should be on the h5 file
            required_keys = [
                "observations/qpos_arm_left",
                "observations/qpos_arm_right",
                "observations/qpos_hand_left",
                "observations/qpos_hand_right",
                "observations/images/aria_rgb_cam/color",
                "observations/images/oakd_front_view/color",
                "actions_arm_left",
                "actions_arm_right",
                "actions_hand_left",
                "actions_hand_right",
            ]

            # Check if all keys are in the h5 file
            for key in required_keys:
                if key in f:
                    data = np.array(f[key])
                    print(f"\n{key}:")
                    print("  shape:", data.shape)
                    print("  dtype:", data.dtype)
                    if data.ndim >= 1 and len(data) > 0:
                        print("  first element preview:", data[0])
                else:
                    print(f"\n⚠️  ALERT: The key {key} is not found inside the h5 file. This will probably make IsaacSim crash.")
                    
    def play_video(self):
        """
        Display an image sequence from the HDF5 file as a video.

        Parameters
        ----------
        video_key : str
            Dataset path of the image sequence inside the HDF5 file.
        delay_ms : int
            Delay between frames in milliseconds.
        """
        video_key="observations/images/aria_rgb_cam/color" # /observations/images/oakd_front_view, observations/images/aria_rgb_cam/color
        delay_ms=30

        try:
            import cv2
        except ImportError:
            print("ERROR: OpenCV not installed.")
            print("Install it with: pip install opencv-python")
            return

        with h5py.File(self.file_path, "r") as f:
            if video_key not in f:
                print(f"Dataset '{video_key}' not found in file.")
                return

            images = f[video_key]
            print(f"Playing video from: {video_key}")
            print(f"Video shape: {images.shape}")

            for i in range(images.shape[0]):
                frame = images[i]

                # Convert RGB -> BGR for OpenCV display
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

                cv2.imshow("Aria RGB Cam (Press ESC to quit)", frame)

                # ESC to quit
                if cv2.waitKey(delay_ms) & 0xFF == 27:
                    break

        cv2.destroyAllWindows()

    def check_frequency(self, dataset_key="observations/qpos_arm_left"):
        """
        Estimate the recording frequency from a timestamp dataset.
        Falls back to checking common timestamp locations in the file.
        """
        with h5py.File(self.file_path, "r") as f:

            # 1. check for top-level timestamps
            possible_ts_keys = [
                "timestamps",
                "observations/timestamps",
                "t",
                "time",
                "observations/time",
            ]

            ts = None
            for key in possible_ts_keys:
                if key in f:
                    ts = np.array(f[key])
                    print(f"Found timestamps at: '{key}'")
                    break

            if ts is None:
                print("No timestamp dataset found. Checking file attributes...")
                print("Top-level attrs:", dict(f.attrs))

                # fall back to estimating from dataset size only
                data = np.array(f[dataset_key])
                print(f"\nDataset '{dataset_key}' has {data.shape[0]} frames.")
                print("Cannot compute Hz without timestamps.")
                print("Ask your supervisor what frequency the robot was controlled at.")
                return

        # compute frequency from timestamps
        diffs = np.diff(ts)
        mean_dt = np.mean(diffs)
        std_dt  = np.std(diffs)
        hz      = 1.0 / mean_dt

        print(f"\nTimestamp stats:")
        print(f"  Total frames : {len(ts)}")
        print(f"  Total duration: {ts[-1] - ts[0]:.2f} s")
        print(f"  Mean dt      : {mean_dt*1000:.2f} ms")
        print(f"  Std  dt      : {std_dt*1000:.2f} ms")
        print(f"  Estimated Hz : {hz:.1f} Hz")

    def check_quat_convention(self, dataset_key="observations/qpos_arm_left", quat_indices=(3, 4, 5, 6)):
        """
        Detect the quaternion convention used in an EEF pose dataset.

        Assumes the dataset has shape (N, 7) where the first 3 columns are
        XYZ position and the last 4 columns are a quaternion. The method
        checks whether the scalar component (w) is at index 3 (wxyz) or
        index 6 (xyzw) by identifying which candidate column has values
        skewed toward +1.0 — the scalar of a unit quaternion representing
        small-to-moderate rotations is always positive and larger than the
        vector components.

        Parameters
        ----------
        dataset_key : str, optional
            HDF5 dataset path to inspect. Defaults to "observations/qpos_arm_left".
        quat_indices : tuple of int, optional
            Indices of the 4 quaternion columns within the dataset row.
            Defaults to (3, 4, 5, 6).

        Prints
        ------
        - First frame raw quaternion values
        - Norm (should be ~1.0 for valid quaternions)
        - Per-column value ranges and means
        - Best guess at convention: wxyz or xyzw
        - Whether Lula/Isaac Sim conversion is needed
        """
        with h5py.File(self.file_path, "r") as f:
            data = np.array(f[dataset_key])

        quats = data[:, list(quat_indices)]  # (N, 4)

        # --- Norm check ---
        norms = np.linalg.norm(quats, axis=1)
        print(f"Dataset:       {dataset_key}")
        print(f"Shape:         {data.shape}")
        print(f"\nFirst frame quaternion (raw indices {quat_indices}):")
        print(f"  values: {quats[0]}")
        print(f"  norm:   {norms[0]:.6f}  (should be ~1.0)")
        print(f"\nNorm across all frames:")
        print(f"  mean: {norms.mean():.6f}  std: {norms.std():.6f}  "
            f"min: {norms.min():.6f}  max: {norms.max():.6f}")

        # --- Per-column statistics ---
        # w is the scalar component: values cluster near +1.0, mean > 0
        # x/y/z are vector components: values centered near 0
        print(f"\nPer-column statistics:")
        print(f"  {'col':<6} {'min':>8} {'max':>8} {'mean':>8} {'std':>8}")
        for i, idx in enumerate(quat_indices):
            col = quats[:, i]
            print(f"  col {idx}:  {col.min():>8.4f} {col.max():>8.4f} "
                f"{col.mean():>8.4f} {col.std():>8.4f}")

        # --- Convention detection ---
        # w column: mean clearly positive, std lower than vector components
        # Use mean of first vs last candidate column to decide
        mean_first = quats[:, 0].mean()   # would be w if wxyz
        mean_last  = quats[:, 3].mean()   # would be w if xyzw

        print("\n--- Convention Detection ---")
        if mean_first > 0.5 and mean_first > mean_last:
            print("✅ Likely convention: wxyz  (w is first, col 3)")
            print("   Lula/Isaac Sim uses xyzw — conversion needed:")
            print("   quat_xyzw = np.array([q[1], q[2], q[3], q[0]])")
        elif mean_last > 0.5 and mean_last > mean_first:
            print("✅ Likely convention: xyzw  (w is last, col 6)")
            print("   Lula/Isaac Sim uses xyzw — NO conversion needed.")
        else:
            print("⚠️  Ambiguous — could not confidently detect convention.")
            print(f"   col 3 mean: {mean_first:.4f} | col 6 mean: {mean_last:.4f}")
            print("   Recommendation: use the FK cross-check method instead.")


if __name__ == "__main__":
    analyzer = H5Analyzer()