from dataclasses import dataclass
from typing import Optional


@dataclass
class VisConfig:
    enabled: bool = True
    show_eef: bool = True           # draw frames at EEF / target position
    show_offset: bool = True        # draw frames lifted 1 m above (unobstructed view)
    video_mode: bool = False        # preset: reduces faded alpha to 0.15 for screen recording
    eef_alpha: Optional[float] = None  # explicit faded alpha; overrides video_mode if set


# Colors as RGBA tuples in [0, 1]
COLOR_RIGHT     = (1.0, 0.0,  0.0,  1.0)  # red
COLOR_LEFT      = (0.0, 0.0,  0.55, 1.0)  # dark blue
COLOR_RIGHT_RAW  = (1.0, 0.55, 0.55, 0.6)  # light red
COLOR_LEFT_RAW   = (0.4, 0.4,  0.85, 0.6)  # light blue
COLOR_RIGHT_FLIP = (0.3, 0.9,  0.3,  0.8)  # green
COLOR_LEFT_FLIP  = (1.0, 0.9,  0.1,  0.8)  # yellow

# Coordinate frame axis colors (XYZ = RGB); faded variants for IK target overlay
COLOR_AXIS_X       = (1.0,  0.15, 0.15, 1.0)
COLOR_AXIS_Y       = (0.15, 1.0,  0.15, 1.0)
COLOR_AXIS_Z       = (0.15, 0.15, 1.0,  1.0)
COLOR_AXIS_X_FADED = (1.0,  0.15, 0.15, 0.35)
COLOR_AXIS_Y_FADED = (0.15, 1.0,  0.15, 0.35)
COLOR_AXIS_Z_FADED = (0.15, 0.15, 1.0,  0.35)

POINT_SIZE      = 10  # pixels
ORIENT_LINE_SIZE = 2   # pixels
ORIENT_TIP_SIZE  = 4   # pixels
FRAME_LINE_SIZE  = 2   # pixels
ORIENT_LENGTH    = 0.1  # metres


class EEFVisualizer:
    def __init__(self):
        from isaacsim.util.debug_draw import _debug_draw
        self._draw = _debug_draw.acquire_debug_draw_interface()

    def clear(self):
        """Clear all accumulated debug lines and points from the previous frame."""
        self._draw.clear_lines()
        self._draw.clear_points()

    def draw_point(self, pos, color):
        """Draw a single point at pos with the given color.

        Args:
            pos:   array-like [x, y, z]
            color: RGBA tuple in [0, 1]
        """
        p = (float(pos[0]), float(pos[1]), float(pos[2]))
        self._draw.draw_points([p], [color], [POINT_SIZE])

    def draw_orientation(self, pos, quat_xyzw, color, length=ORIENT_LENGTH):
        """Draw an orientation indicator: a line from pos along local Z, with a dot at the tip.

        Args:
            pos:       array-like [x, y, z]
            quat_xyzw: unit quaternion as [x, y, z, w]
            color:     RGBA tuple in [0, 1]
            length:    line length in metres
        """
        x = float(quat_xyzw[0])
        y = float(quat_xyzw[1])
        z = float(quat_xyzw[2])
        w = float(quat_xyzw[3])

        # Rotate local Z axis [0,0,1] by the quaternion (Rodrigues formula)
        dir_x = 2.0 * (w * y + x * z)
        dir_y = 2.0 * (y * z - w * x)
        dir_z = 1.0 - 2.0 * (x * x + y * y)

        p0 = (float(pos[0]),                      float(pos[1]),                      float(pos[2]))
        p1 = (float(pos[0]) + dir_x * length,     float(pos[1]) + dir_y * length,     float(pos[2]) + dir_z * length)

        self._draw.draw_lines([p0], [p1], [color], [float(ORIENT_LINE_SIZE)])
        self._draw.draw_points([p1],    [color], [ORIENT_TIP_SIZE])

    def draw_frame(self, pos, quat_xyzw, color_x, color_y, color_z, length=ORIENT_LENGTH, width=FRAME_LINE_SIZE):
        """Draw a coordinate frame (3 axis lines) at pos oriented by quat_xyzw.

        Args:
            pos:         array-like [x, y, z]
            quat_xyzw:   unit quaternion as [x, y, z, w]
            color_x/y/z: RGBA tuples for each axis
            length:      axis line length in metres
            width:       line width in pixels
        """
        x = float(quat_xyzw[0])
        y = float(quat_xyzw[1])
        z = float(quat_xyzw[2])
        w = float(quat_xyzw[3])

        # Rotate each unit axis by the quaternion (rotation matrix columns)
        ax = (1 - 2*(y*y + z*z),     2*(x*y + w*z),     2*(x*z - w*y))
        ay = (    2*(x*y - w*z), 1 - 2*(x*x + z*z),     2*(y*z + w*x))
        az = (    2*(x*z + w*y),     2*(y*z - w*x), 1 - 2*(x*x + y*y))

        p0 = (float(pos[0]), float(pos[1]), float(pos[2]))
        px = (p0[0] + ax[0]*length, p0[1] + ax[1]*length, p0[2] + ax[2]*length)
        py = (p0[0] + ay[0]*length, p0[1] + ay[1]*length, p0[2] + ay[2]*length)
        pz = (p0[0] + az[0]*length, p0[1] + az[1]*length, p0[2] + az[2]*length)

        self._draw.draw_lines([p0], [px], [color_x], [float(width)])
        self._draw.draw_lines([p0], [py], [color_y], [float(width)])
        self._draw.draw_lines([p0], [pz], [color_z], [float(width)])