"""
quat_utils.py

Quaternion utility functions.

Used by simulator.py
"""
import numpy as np
from scipy.spatial.transform import Rotation

Q_TOOL_TO_URDF_WXYZ = np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32)  # Rx(180°)

def xyzw_to_wxyz(q: np.ndarray) -> np.ndarray:
    return np.array([q[3], q[0], q[1], q[2]], dtype=np.float32)

def wxyz_to_xyzw(q: np.ndarray) -> np.ndarray:
    return np.array([q[1], q[2], q[3], q[0]], dtype=np.float32)


def normalize_quat_wxyz(quat):
    quat = np.asarray(quat, dtype=np.float32).reshape(4)
    n = np.linalg.norm(quat)
    if n < 1e-8:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    return quat / n

def quat_multiply_wxyz(q1, q2):
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    ], dtype=np.float32)


def _sensor_to_robot_frame(q_wxyz):
    """Remap quaternion from sensor convention (LH, X-forward) to robot URDF convention (RH, Z-forward)."""
    w, x, y, z = q_wxyz
    # Combined effect: negate x,z (LH→RH handedness) then swap x↔z (axis relabeling)
    return np.array([w, -z, y, -x], dtype=np.float32)


def tool_quat_to_urdf(q_tool_wxyz):
    q = normalize_quat_wxyz(q_tool_wxyz)
    q = _sensor_to_robot_frame(q)
    return normalize_quat_wxyz(quat_multiply_wxyz(Q_TOOL_TO_URDF_WXYZ, q))


def wxyz_to_rotation_matrix(q):
    w, x, y, z = normalize_quat_wxyz(q)
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ], dtype=np.float32)

def parse_ori_correction(spec: str) -> np.ndarray:
    """Parse a rotation spec into a 3x3 matrix.

    Formats: "none" | "axis:degrees" | "axis:degrees, axis:degrees"
    (e.g. "y:-90", "y:-90, x:45")
    """
    if spec is None or spec == "none":
        return np.eye(3, dtype=np.float32)

    correction = np.eye(3, dtype=np.float32)
    for part in str(spec).split(","):
        part = part.strip()
        try:
            axis, deg = part.split(":", maxsplit=1)
            axis = axis.strip().lower()
            angle = float(deg)
        except ValueError:
            raise ValueError(
                f"Invalid grasp_eef_ori_flip '{spec}'. Use 'none' or comma-separated "
                "axis:degrees entries (e.g. 'y:-90' or 'y:-90, x:45')."
            )
        if axis not in ("x", "y", "z"):
            raise ValueError(f"Invalid axis '{axis}' in grasp_eef_ori_flip. Must be x, y, or z.")
        correction = correction @ Rotation.from_euler(axis, angle, degrees=True).as_matrix().astype(np.float32)
    return correction


def detect_quaternion_order(arm_data, label):
    w_if_wxyz = float(np.mean(np.abs(arm_data[:, 3])))
    w_if_xyzw = float(np.mean(np.abs(arm_data[:, 6])))
    print(f"[quat] {label}: mean|col3|={w_if_wxyz:.4f} mean|col6|={w_if_xyzw:.4f}")
    if w_if_xyzw > w_if_wxyz:
        print(f"[quat] {label}: detected xyzw ordering. Reordering to wxyz.")
        reordered = arm_data.copy()
        reordered[:, 3] = arm_data[:, 6]
        reordered[:, 4] = arm_data[:, 3]
        reordered[:, 5] = arm_data[:, 4]
        reordered[:, 6] = arm_data[:, 5]
        return reordered
    print(f"[quat] {label}: appears to be wxyz ordering. Using as-is.")
    return arm_data
