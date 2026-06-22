"""
IK solver class

solver
EE_FRAME_NAME
EE_FLANGE_TO_EEF_OFFSET
PANDA_ARM_DESCRIPTION_PATH
PANDA_ARM_URDF_PATH
"""
from isaacsim.robot_motion.motion_generation import LulaKinematicsSolver
import numpy as np
from src.simulator.quat_utils import wxyz_to_rotation_matrix

class FrankaIKController:
    def __init__(
        self,
        label,
        robot_description_path,
        urdf_path,
        ee_frame_name,
        flange_to_eef_offset,
    ):
        self.label = label
        self.ee_frame_name = ee_frame_name
        self.flange_to_eef_offset = flange_to_eef_offset

        self.solver = LulaKinematicsSolver(
            robot_description_path=robot_description_path,
            urdf_path=urdf_path,
        )

        print(f"[IK] Solver ({label}) created.")
        print(f"[IK] Active joints: {self.solver.get_joint_names()}")
        print(f"[IK] Available frames: {self.solver.get_all_frame_names()}")

    def compute(self, target_wrist_pos, target_quat_wxyz, warm_start=None):
        rot = wxyz_to_rotation_matrix(target_quat_wxyz)
        ik_position = (
            np.asarray(target_wrist_pos, dtype=np.float32)
            - rot @ self.flange_to_eef_offset
        )

        joint_positions, success = self.solver.compute_inverse_kinematics(
            frame_name=self.ee_frame_name,
            target_position=ik_position,
            target_orientation=target_quat_wxyz,
            warm_start=warm_start,
        )

        if success:
            return np.asarray(joint_positions, dtype=np.float32), True

        return None, False