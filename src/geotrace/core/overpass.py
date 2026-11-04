"""Overpass API client with bounding-box, timeout and result limits."""

import math

import httpx

ENDPOINT = "https://overpass-api.de/api/interpreter"
USER_AGENT = "GeoTrace/0.1.0"

BBOX_PLACEHOLDER = "{{bbox}}"

MAX_BBOX_KM2 = 100.0
DEFAULT_TIMEOUT_S = 25
MAX_TIMEOUT_S = 90
DEFAULT_MAX_RESULTS = 100
MAX_RESULTS_LIMIT = 1000

EARTH_RADIUS_KM = 6371.0088
# Extra seconds on the HTTP client beyond the server-side timeout, for transfer.
HTTP_MARGIN_S = 10


class OverpassError(ValueError):
    """Raised for invalid input or a failed request. The message is safe to show the model."""


def bbox_area_km2(south: float, west: float, north: float, east: float) -> float:
    """Approximate area of a lat/lon box on a sphere, in km²."""
    lon_span = math.radians(east - west)
    return (EARTH_RADIUS_KM ** 2) * lon_span * abs(
        math.sin(math.radians(north)) - math.sin(math.radians(south))
    )


def validate_bbox(south: float, west: float, north: float, east: float) -> float:
    """Check the box and return its area in km². Raises OverpassError if it is not allowed."""
    for name, value in (("south", south), ("west", west), ("north", north), ("east", east)):
        if not math.isfinite(value):
            raise OverpassError(f"{name} must be a finite number")
    if not -90 <= south < north <= 90:
        raise OverpassError("need -90 <= south < north <= 90")
    if not -180 <= west < east <= 180:
        raise OverpassError("need -180 <= west < east <= 180 (boxes crossing the antimeridian are not supported)")
    area = bbox_area_km2(south, west, north, east)
    if area > MAX_BBOX_KM2:
        raise OverpassError(
            f"bounding box is {area:.1f} km², above the maximum of {MAX_BBOX_KM2:g} km². Use a smaller box."
        )
    return area


def build_query(query: str, south: float, west: float, north: float, east: float, timeout: int) -> str:
    """Wrap the user's Overpass QL with the output and timeout settings and substitute the bbox.

    The query must contain the {{bbox}} placeholder, so the box limit always applies.
    """
    if BBOX_PLACEHOLDER not in query:
        raise OverpassError(
            f"query must contain the {BBOX_PLACEHOLDER} placeholder in each filter, "
            "for example node[\"amenity\"=\"cafe\"]({{bbox}});"
        )
    bbox = f"{south},{west},{north},{east}"
    body = query.replace(BBOX_PLACEHOLDER, bbox)
    return f"[out:json][timeout:{timeout}];\n{body}\nout body;"


async def run_query(
    query: str,
    south: float,
    west: float,
    north: float,
    east: float,
    timeout: int = DEFAULT_TIMEOUT_S,
    max_results: int = DEFAULT_MAX_RESULTS,
) -> dict:
    if not query or not query.strip():
        raise OverpassError("query must not be empty")
    if not 1 <= timeout <= MAX_TIMEOUT_S:
        raise OverpassError(f"timeout must be between 1 and {MAX_TIMEOUT_S} seconds")
    if not 1 <= max_results <= MAX_RESULTS_LIMIT:
        raise OverpassError(f"max_results must be between 1 and {MAX_RESULTS_LIMIT}")

    area = validate_bbox(south, west, north, east)
    full_query = build_query(query, south, west, north, east, timeout)

    try:
        async with httpx.AsyncClient(
            timeout=httpx.Timeout(timeout + HTTP_MARGIN_S),
            headers={"User-Agent": USER_AGENT},
        ) as client:
            response = await client.post(ENDPOINT, data={"data": full_query})
    except httpx.TimeoutException as exc:
        raise OverpassError(f"Overpass request timed out after {timeout} seconds") from exc
    except httpx.HTTPError as exc:
        raise OverpassError(f"Overpass request failed: {exc}") from exc

    if response.status_code != 200:
        detail = response.text.strip()[:500]
        raise OverpassError(f"Overpass returned HTTP {response.status_code}: {detail}")

    try:
        payload = response.json()
    except ValueError as exc:
        raise OverpassError("Overpass returned a response that is not JSON") from exc

    elements = payload.get("elements", [])
    total = len(elements)
    returned = elements[:max_results]

    result = {
        "bbox_area_km2": round(area, 3),
        "count_total": total,
        "count_returned": len(returned),
        "truncated": total > max_results,
        "elements": returned,
    }
    if "remark" in payload:
        # Overpass reports runtime errors, such as a server-side timeout, in "remark".
        result["remark"] = payload["remark"]
    return result
