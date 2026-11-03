import math
from dataclasses import dataclass

Point = tuple[float, float]
Polygons = list[list[list[int]]]  # as returned by the Space: [[[x, y], ...], ...]


@dataclass
class ShadowGeometry:
    pair_score: float            # model confidence (min between object and shadow)
    shadow_direction_deg: float  # on the ground, clockwise from the camera's forward direction (0 = away, 90 = right)
    sun_elevation_deg: float     # atan(height / length), calibrated if a camera model is given


def _points(polygons: Polygons) -> list[Point]:
    return [(float(x), float(y)) for poly in polygons for x, y in poly]


def _extreme(points: list[Point], highest: bool) -> Point:
    """Highest (min y) or lowest (max y) point; on ties, the average x."""
    y = min(p[1] for p in points) if highest else max(p[1] for p in points)
    xs = [p[0] for p in points if p[1] == y]
    return (sum(xs) / len(xs), y)


def sun_elevation_calibration(top_px, base_px, tip_px, K, gravity) -> float:
    """Sun elevation (degrees) from 3 pixels and the camera calibration.

    K: 3x3 intrinsic matrix (focal and centre of the image).
    gravity: unit vector pointing down, in camera coordinates (x right, y down, z forward).
    The camera height is set to 1: the scale cancels out in the ratio."""
    import numpy as np

    K_inv = np.linalg.inv(K)
    g = gravity / np.linalg.norm(gravity)

    def ray(px):
        return K_inv @ np.array([px[0], px[1], 1.0])

    def on_ground(px):
        d = ray(px)
        t = 1.0 / (d @ g)               # the ground is at distance 1 "downwards"
        if t <= 0:
            raise ValueError("Point is above the horizon: it can't be on the ground")
        return t * d

    base, tip = on_ground(base_px), on_ground(tip_px)
    # top = base - H*g must lie on the ray of the top pixel: s*ray = base - H*g
    (_, height), *_ = np.linalg.lstsq(
        np.column_stack([ray(top_px), g]), base, rcond=None
    )
    length = np.linalg.norm(tip - base)
    return float(np.degrees(np.arctan2(height, length)))


def shadow_direction_calibration(base_px, tip_px, K, gravity) -> float:
    """Direction of the shadow on the ground, clockwise seen from above, measured from the
    camera's forward direction (0 = away from the camera, 90 = to the right).

    Measured on the ground plane, so it does not change with the camera roll.
    Uses the same calibrated ground points as sun_elevation_calibrated."""
    import numpy as np

    K_inv = np.linalg.inv(K)
    g = gravity / np.linalg.norm(gravity)

    def on_ground(px):
        d = K_inv @ np.array([px[0], px[1], 1.0])
        return d / (d @ g)

    def horizontal(v):
        return v - (v @ g) * g

    shadow = horizontal(on_ground(tip_px) - on_ground(base_px))
    forward = horizontal(np.array([0.0, 0.0, 1.0]))
    # Horizontal right from gravity and forward. The camera's x axis is not used:
    # it tilts with the roll, so it would make the angle depend on the roll.
    right = np.cross(g, forward)
    angle = math.degrees(math.atan2(shadow @ right, shadow @ forward))
    return angle % 360


def analyze_pair(object_polygons: Polygons,
                 shadow_polygons: Polygons,
                 obj_score: float,
                 sha_score: float,
                 K,
                 gravity) -> ShadowGeometry:
    """Shadow geometry from one object-shadow pair, with the sun elevation computed
    from the camera calibration (see calibration.calibrate)."""
    obj, sha = _points(object_polygons), _points(shadow_polygons)
    if not obj or not sha:
        raise ValueError("The pair needs both an object and a shadow")

    top = _extreme(obj, highest=True)
    base = _extreme(obj, highest=False)
    # The farthest point of a polygon from a given point is always a vertex
    tip = max(sha, key=lambda p: math.dist(p, base))

    sun_direction = (shadow_direction_calibration(base, tip, K, gravity) + 180) % 360
    sun_altitude = sun_elevation_calibration(top, base, tip, K, gravity)

    return ShadowGeometry(
        pair_score=round(min(obj_score, sha_score), 3),
        sun_relative_azimuth_to_camera_deg=round(sun_direction, 1),
        sun_altitude_deg=round(sun_altitude, 1),
    )
