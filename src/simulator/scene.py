"""
scene.py

USD and articulation utility functions used by simulator.py.
"""

import numpy as np
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from isaacsim.core.prims import XFormPrim


@dataclass
class ActiveObject:
    name: str
    prim_path: str
    prim: "XFormPrim"
    asset_path: Path | None
    traj: np.ndarray
    state: np.ndarray | None
    grasp_eef_side: str
    grasp_eef_offset: np.ndarray
    grasp_ori_correction: np.ndarray
    # grasp-anchor fields (populated when object_grasp_anchor is enabled)
    first_grasped_frame: int | None = None
    last_grasped_frame: int | None = None
    pre_grasp_pos: np.ndarray | None = None   # (3,) world position: x,y from first grasped frame, z from physics settling
    pre_grasp_quat: np.ndarray | None = None  # (4,) wxyz orientation from frame just before grasp begins
    grasp_offset_local: np.ndarray | None = None  # (3,) object position in hand-palm local frame, captured at grasp start
    grasp_ori_local: np.ndarray | None = None     # (3,3) object orientation in hand-palm local frame, captured at grasp start
    grasp_active: bool = False                    # True after the dynamic grasp transform has been captured
    physics_release_pending: bool = False      # True after release is requested but before collisions/dynamics are active
    physics_release_frame: int | None = None
    physics_activation_frame: int | None = None
    physics_released: bool = False             # True once kinematic is disabled post-grasp
    collision_mesh_paths: list = None          # USD paths of mesh children with CollisionAPI
    position_offset_world: np.ndarray | None = None  # (3,) runtime world translation applied to replayed poses

logger = logging.getLogger(__name__)

ARM_JOINT_NAMES = [f"panda_joint{i}" for i in range(1, 8)]

HAND_LEFT_JOINT_NAMES = [
    "left_wrist",
    "left_thumb_mcp", "left_thumb_abd", "left_thumb_pip", "left_thumb_dip",
    "left_index_abd", "left_index_mcp", "left_index_pip",
    "left_middle_abd", "left_middle_mcp", "left_middle_pip",
    "left_ring_abd", "left_ring_mcp", "left_ring_pip",
    "left_pinky_abd", "left_pinky_mcp", "left_pinky_pip",
]
HAND_RIGHT_JOINT_NAMES = [
    "right_wrist",
    "right_thumb_mcp", "right_thumb_abd", "right_thumb_pip", "right_thumb_dip",
    "right_index_abd", "right_index_mcp", "right_index_pip",
    "right_middle_abd", "right_middle_mcp", "right_middle_pip",
    "right_ring_abd", "right_ring_mcp", "right_ring_pip",
    "right_pinky_abd", "right_pinky_mcp", "right_pinky_pip",
]


def _get_dof_names(articulation) -> list[str]:
    try:
        return list(articulation.dof_names)
    except Exception:
        return list(articulation.get_dof_names())


def print_articulation_info(articulation, label):
    dof_names = _get_dof_names(articulation)
    logger.info("[scene] %s — %d DOFs", label, len(dof_names))
    for i, name in enumerate(dof_names):
        logger.debug("[scene]   [%02d] %s", i, name)


def resolve_dof_indices(articulation, names, label):
    dof_names = _get_dof_names(articulation)
    name_to_idx = {n: i for i, n in enumerate(dof_names)}

    def candidates(name):
        if name.startswith("panda_joint"):
            return [name, name.replace("panda_joint", "fer_joint", 1)]
        if name.startswith("fer_joint"):
            return [name, name.replace("fer_joint", "panda_joint", 1)]
        return [name]

    indices = []
    for name in names:
        resolved = False
        for cand in candidates(name):
            if cand in name_to_idx:
                if cand != name:
                    logger.debug("[scene] DOF '%s' matched via alias '%s'", name, cand)
                indices.append(name_to_idx[cand])
                resolved = True
                break
        if resolved:
            continue

        matches = [dof for dof in dof_names if dof.endswith(name)]
        if len(matches) == 1:
            logger.debug("[scene] DOF '%s' matched via suffix to '%s'", name, matches[0])
            indices.append(name_to_idx[matches[0]])
        elif len(matches) > 1:
            raise RuntimeError(f"Ambiguous suffix match for '{name}' in {label}: {matches}")
        else:
            raise RuntimeError(f"Cannot find '{name}' in {label} DOFs: {dof_names}")

    return np.array(indices, dtype=int)


def resolve_descendant_prim_path(stage, subtree_root, prim_name):
    direct_path = f"{subtree_root}/{prim_name}"
    if stage.GetPrimAtPath(direct_path).IsValid():
        return direct_path

    matches = [
        str(prim.GetPath())
        for prim in stage.Traverse()
        if str(prim.GetPath()).startswith(f"{subtree_root}/") and prim.GetName() == prim_name
    ]

    if len(matches) == 1:
        logger.debug("[scene] Resolved '%s' under '%s' -> %s", prim_name, subtree_root, matches[0])
        return matches[0]

    if not matches:
        raise RuntimeError(
            f"Could not find prim '{prim_name}' under '{subtree_root}' in the loaded USD stage"
        )

    raise RuntimeError(
        f"Ambiguous prim '{prim_name}' under '{subtree_root}': {matches}"
    )
