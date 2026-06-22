import numpy as np
from scipy.spatial.transform import Rotation as R

# Given transformation, from robot base
best_calib = {
    "left_cam": np.array([
        [+0.09508648, -0.87832947, +0.46850389, +0.14624048],
        [-0.96130139, +0.04122833, +0.27239652, -0.41912299],
        [-0.25856952, -0.47627467, -0.84041909, +0.69073473],
        [ 0.        ,  0.        ,  0.        ,  1.        ]
    ]),
    "right_cam": np.array([
        [ 0.02933941, -0.83227828,  0.55358113,  0.17515134],
        [-0.99642232,  0.01956109,  0.08221870,  0.34649483],
        [-0.07925749, -0.55401284, -0.82872675,  0.46895363],
        [ 0.        ,  0.        ,  0.        ,  1.        ]
    ])
}

# Transformation -> robot base
T_base_left_world = np.array([
    [1.0, 0.0, 0.0, -0.25],
    [0.0, 1.0, 0.0,  0.35],
    [0.0, 0.0, 1.0,  0.77],
    [0.0, 0.0, 0.0,  1.0],
])

T_base_right_world = np.array([
    [1.0, 0.0, 0.0, -0.25],
    [0.0, 1.0, 0.0, -0.35],
    [0.0, 0.0, 1.0,  0.77],
    [0.0, 0.0, 0.0,  1.0],
])

R_fix = np.diag([1.0, -1.0, -1.0])

def apply_rotation_fix(T, R_fix):
    T_new = T.copy()
    T_new[:3, :3] = T[:3, :3] @ R_fix
    return T_new

def transform_to_xyz_translation(T, degrees=True):
    T = np.asarray(T)

    translation = T[:3, 3]
    rotation_matrix = T[:3, :3]
    euler_xyz = R.from_matrix(rotation_matrix).as_euler("xyz", degrees=degrees)

    return translation, euler_xyz

def format_tuple(arr, precision=6):
    return f"({arr[0]:.{precision}f}, {arr[1]:.{precision}f}, {arr[2]:.{precision}f})"

# Transformations - world base
T_cam_left_world  = T_base_left_world @ best_calib["left_cam"]
T_cam_right_world = T_base_right_world @ best_calib["right_cam"]

# Apply final rotation fix
T_cam_left_world_fixed  = apply_rotation_fix(T_cam_left_world, R_fix)
T_cam_right_world_fixed = apply_rotation_fix(T_cam_right_world, R_fix)

t1, r1 = transform_to_xyz_translation(T_cam_left_world_fixed, degrees=True)
t2, r2 = transform_to_xyz_translation(T_cam_right_world_fixed, degrees=True)

print("Translation 1:", format_tuple(t1))
print("Rotation XYZ 1 (deg):", format_tuple(r1))

print("Translation 2:", format_tuple(t2))
print("Rotation XYZ 2 (deg):", format_tuple(r2))