"""Build the dual-arm scene from scratch.

Creates a new USD stage, adds two independent tables with calibrated cameras,
references one robot on each table, places the manipulation objects, then
saves to ./scenes/scene.usd.
"""

from isaacsim import SimulationApp
VISUALIZE = True
simulation_app = SimulationApp({"headless": not VISUALIZE})

from pathlib import Path
import os
import numpy as np
import omni.usd
from pxr import UsdGeom, UsdLux, UsdPhysics, Gf

BASE_DIR = Path(__file__).resolve().parent.parent
DESCRIPTION_ROOT = Path(
    os.environ.get(
        "PANDAORCA_ROOT",
        str(BASE_DIR / "assets" / "pandaorca_description-main"),
    )
)

LEFT_ROBOT_USD  = DESCRIPTION_ROOT / "usd" / "fer_orcahand_left_extended"  / "fer_orcahand_left_extended.usd"
RIGHT_ROBOT_USD = DESCRIPTION_ROOT / "usd" / "fer_orcahand_right_extended" / "fer_orcahand_right_extended.usd"
OUTPUT_SCENE    = BASE_DIR / "scenes" / "scene.usd"

# Cameras
# T/R values for USD camera prims. Rotations already include the OpenCV camera
# frame -> USD camera frame conversion: R_usd = R_opencv @ diag(1, -1, -1).
T1 = (-0.091753, 0.064681, 1.209793)
R1 = (39.046318, 0.774982, -91.352988)
T2 = (0.046454, -0.024522, 1.412295)
R2 = (23.823474, -0.050316, -92.367329)


ARIA_INTRINSICS = np.array([
    [133.25430222 * 2, 0.0, 320, 0],
    [0.0, 133.25430222 * 2, 240, 0],
    [0.0, 0.0, 1.0, 0]
])

ARIA_INTRINSICS_HALF = np.array([
    [133.25430222, 0.0, 320 / 2, 0],
    [0.0, 133.25430222, 240 / 2, 0],
    [0.0, 0.0, 1.0, 0],
])

# OAK-D front-view intrinsics from the stereo H5, scaled to 960×540.
# Using the correct K here affects the rendered/framed camera view even when
# the camera world pose is already correct.
OAKD_INTRINSICS = np.array([
    [598.47614, 0.0, 488.64874, 0],
    [0.0, 598.70917, 276.32925, 0],
    [0.0, 0.0, 1.0, 0],
])

ARIA_IMAGE_SIZE = (640, 480)
OAKD_IMAGE_SIZE = (960, 540)

def add_reference(stage, prim_path: str, asset_path: Path):
    prim = stage.DefinePrim(prim_path, "Xform")
    prim.GetReferences().AddReference(str(asset_path))
    print(f"Added reference: {prim_path} -> {asset_path}")
    return prim

def set_xform(prim, translate=(0, 0, 0), rotate_xyz=(0, 0, 0), scale=(1, 1, 1)):
    xform = UsdGeom.Xformable(prim)
    xform.AddTranslateOp(opSuffix="scene").Set(Gf.Vec3d(*translate))
    xform.AddRotateXYZOp(opSuffix="scene").Set(Gf.Vec3f(*rotate_xyz))
    xform.AddScaleOp(opSuffix="scene").Set(Gf.Vec3f(*scale))
    print(f"  translate={translate}  rotateXYZ={rotate_xyz}  scale={scale}")


def set_xform_from_matrix(prim, T):
    """Set prim transform from a 4×4 homogeneous matrix (prim relative to parent)."""
    T = np.asarray(T, dtype=np.float64)
    xform = UsdGeom.Xformable(prim)
    xform.ClearXformOpOrder()
    xform.AddTransformOp(opSuffix="scene").Set(
        Gf.Matrix4d(
            T[0,0], T[0,1], T[0,2], T[0,3],
            T[1,0], T[1,1], T[1,2], T[1,3],
            T[2,0], T[2,1], T[2,2], T[2,3],
            T[3,0], T[3,1], T[3,2], T[3,3],
        )
    )

def create_table(stage, prim_path: str, translate, scale):
    prim = stage.DefinePrim(prim_path, "Cube")
    UsdPhysics.CollisionAPI.Apply(prim)
    set_xform(prim, translate=translate, scale=scale)
    print(f"Table created: {prim_path}")
    return prim

def add_camera(stage, prim_path: str, intrinsics: np.ndarray, translate, rotate_xyz=(0, 0, 0), image_size=(640, 480), clipping_range=(0.01, 100.0), horizontal_aperture_mm=20.955,):
    """Overhead camera"""
    intrinsics = np.asarray(intrinsics, dtype=np.float64)
    if intrinsics.shape not in [(3, 3), (3, 4)]:
        raise ValueError(f"Expected intrinsics shape (3,3) or (3,4), got {intrinsics.shape}")

    width, height = image_size
    fx = float(intrinsics[0, 0])
    fy = float(intrinsics[1, 1])
    cx = float(intrinsics[0, 2])
    cy = float(intrinsics[1, 2])

    focal_length_mm = fx * horizontal_aperture_mm / width
    vertical_aperture_mm = horizontal_aperture_mm * height / width


    # Principal point offsets from image center - USD offsets are in mm on the sensor plane.
    horiz_offset_mm = (cx - width / 2.0) * horizontal_aperture_mm / width
    vert_offset_mm = (cy - height / 2.0) * vertical_aperture_mm / height

    cam = UsdGeom.Camera.Define(stage, prim_path)
    xform = UsdGeom.Xformable(cam)
    xform.AddTranslateOp(opSuffix="scene").Set(Gf.Vec3d(*translate))
    xform.AddRotateXYZOp(opSuffix="scene").Set(Gf.Vec3f(*rotate_xyz))
    cam.CreateHorizontalApertureAttr(horizontal_aperture_mm)
    cam.CreateVerticalApertureAttr(vertical_aperture_mm)
    cam.CreateFocalLengthAttr(focal_length_mm)
    cam.CreateClippingRangeAttr(Gf.Vec2f(*clipping_range))
    cam.CreateHorizontalApertureOffsetAttr(horiz_offset_mm)
    cam.CreateVerticalApertureOffsetAttr(vert_offset_mm)

    print(f"Camera created: {prim_path}  translate={translate}  image_size={image_size}")
    return cam

def main():
    ctx = omni.usd.get_context()
    ctx.new_stage()
    stage = ctx.get_stage()
    if stage is None:
        raise RuntimeError("Could not create a new USD stage.")

    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)

    stage.DefinePrim("/World", "Xform")

    # Physics
    physics = UsdPhysics.Scene.Define(stage, "/World/PhysicsScene")
    physics.CreateGravityDirectionAttr(Gf.Vec3f(0, 0, -1))
    physics.CreateGravityMagnitudeAttr(9.81)

    # Ground plane
    ground = UsdGeom.Mesh.Define(stage, "/World/GroundPlane/CollisionMesh")
    ground.CreatePointsAttr([(-25, -25, 0), (25, -25, 0), (25, 25, 0), (-25, 25, 0)])
    ground.CreateFaceVertexCountsAttr([4])
    ground.CreateFaceVertexIndicesAttr([0, 1, 2, 3])
    ground.CreateDisplayColorAttr([(0.5, 0.5, 0.5)])
    UsdPhysics.CollisionAPI.Apply(
        stage.DefinePrim("/World/GroundPlane/CollisionPlane", "Plane")
    )

    # Lighting
    dome = UsdLux.DomeLight.Define(stage, "/World/DomeLight")
    dome.CreateIntensityAttr(1000.0)
    dome.CreateColorAttr(Gf.Vec3f(1.0, 0.98, 0.95))

    # ── LEFT SIDE ──────────────────────────────────────────────────────────
    create_table(
        stage, "/World/table_left",
        translate=(0.0, 0.35, 0.75),
        scale=(0.5, 0.35, 0.02),
    )
    # Overhead camera above the left table, looking straight down
    add_camera(
        stage, "/World/camera_left",
        intrinsics=ARIA_INTRINSICS,
        translate=T1,
        rotate_xyz=R1,
        image_size=ARIA_IMAGE_SIZE,
    )

    # Robot
    if LEFT_ROBOT_USD.exists():
        left = add_reference(stage, "/World/fer_orcahand_left_extended", LEFT_ROBOT_USD)
        left.SetInstanceable(False)
        set_xform(left, translate=(-0.255, 0.35, 0.77))
    else:
        print(f"WARNING: left robot USD not found: {LEFT_ROBOT_USD}")

    # ── RIGHT SIDE ─────────────────────────────────────────────────────────
    create_table(
        stage, "/World/table_right",
        translate=(0.0, -0.35, 0.75),
        scale=(0.5, 0.35, 0.02),
    )
    add_camera(
        stage, "/World/camera_right",
        intrinsics=OAKD_INTRINSICS,
        translate=T2,
        rotate_xyz=R2,
        image_size=OAKD_IMAGE_SIZE,
    )

    if RIGHT_ROBOT_USD.exists():
        right = add_reference(stage, "/World/fer_orcahand_right_extended", RIGHT_ROBOT_USD)
        right.SetInstanceable(False)
        set_xform(right, translate=(-0.255, -0.35, 0.77))
    else:
        print(f"WARNING: right robot USD not found: {RIGHT_ROBOT_USD}")

    # Save
    OUTPUT_SCENE.parent.mkdir(parents=True, exist_ok=True)
    if not ctx.save_as_stage(str(OUTPUT_SCENE)):
        raise RuntimeError(f"Could not save scene to: {OUTPUT_SCENE}")
    print(f"\nScene saved to: {OUTPUT_SCENE}")

    # Visualization loop runs after save so the scene is always written correctly
    if VISUALIZE:
        print("Visualization mode active. Close the window or press Ctrl+C to exit.")
        try:
            while simulation_app.is_running():
                simulation_app.update()
        except KeyboardInterrupt:
            pass

    simulation_app.close()


if __name__ == "__main__":
    main()
