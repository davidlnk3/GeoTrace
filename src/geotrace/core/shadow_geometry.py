import math
from dataclasses import dataclass

Point = tuple[float, float]
Polygons = list[list[list[int]]]  # as returned by the Space: [[[x, y], ...], ...]


@dataclass
class ShadowGeometry:
    pair_score: float            # model confidence (min between object and shadow)
    shadow_direction_deg: float  # in the image, clockwise from "up" (0 = up, 90 = right)
    sun_elevation_deg: float     # atan(height / length), calibrated if a camera model is given


def _points(polygons: Polygons) -> list[Point]:
    return [(float(x), float(y)) for poly in polygons for x, y in poly]


def _extreme(points: list[Point], highest: bool) -> Point:
    """Highest (min y) or lowest (max y) point; on ties, the average x."""
    y = min(p[1] for p in points) if highest else max(p[1] for p in points)
    xs = [p[0] for p in points if p[1] == y]
    return (sum(xs) / len(xs), y)


def sun_elevation_calibrated(top_px, base_px, tip_px, K, gravity) -> float:
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

    dx, dy = tip[0] - base[0], tip[1] - base[1]
    direction = math.degrees(math.atan2(dx, -dy)) % 360

    elevation = sun_elevation_calibrated(top, base, tip, K, gravity)

    return ShadowGeometry(
        pair_score=round(min(obj_score, sha_score), 3),
        shadow_direction_deg=round(direction, 1),
        sun_elevation_deg=round(elevation, 1),
    )
