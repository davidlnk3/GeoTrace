import math
from dataclasses import dataclass

Point = tuple[float, float]
Polygons = list[list[list[int]]]  # as returned by the Space: [[[x, y], ...], ...]


@dataclass
class ShadowGeometry:
    pair_score: float            # model confidence (min between object and shadow)
    shadow_direction_deg: float  # in the image, clockwise from "up" (0 = up, 90 = right)
    estimated_sun_elevation_deg: float  # atan(height / length); approximate


def _points(polygons: Polygons) -> list[Point]:
    return [(float(x), float(y)) for poly in polygons for x, y in poly]


def _extreme(points: list[Point], highest: bool) -> Point:
    """Highest (min y) or lowest (max y) point; on ties, the average x."""
    y = min(p[1] for p in points) if highest else max(p[1] for p in points)
    xs = [p[0] for p in points if p[1] == y]
    return (sum(xs) / len(xs), y)


def analyze_pair(object_polygons: Polygons, 
                 shadow_polygons: Polygons,
                 obj_score: float,
                 sha_score: float) -> ShadowGeometry:
    obj, sha = _points(object_polygons), _points(shadow_polygons)
    if not obj or not sha:
        raise ValueError("The pair needs both an object and a shadow")

    top = _extreme(obj, highest=True)
    base = _extreme(obj, highest=False)
    # The farthest point of a polygon from a given point is always a vertex
    tip = max(sha, key=lambda p: math.dist(p, base))

    height = math.dist(top, base)
    length = math.dist(base, tip)
    dx, dy = tip[0] - base[0], tip[1] - base[1]
    direction = math.degrees(math.atan2(dx, -dy)) % 360

    return ShadowGeometry(
        pair_score=round(min(obj_score, sha_score), 3),
        shadow_direction_deg=round(direction, 1),
        estimated_sun_elevation_deg=round(math.degrees(math.atan2(height, length)), 1),
    )