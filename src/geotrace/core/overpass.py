"""Overpass API client with bounding-box, timeout and result limits."""

import math
import re
import time

import httpx

# Tried in order. Each later endpoint is used only when the earlier ones fail.
ENDPOINTS = [
    "https://overpass-api.de/api/interpreter",
    "https://overpass.kumi.systems/api/interpreter",
    "https://overpass.private.coffee/api/interpreter",
    "https://maps.mail.ru/osm/tools/overpass/api/interpreter",
    "https://ovp.dresel.systems/api/interpreter",
]
USER_AGENT = "GeoTrace/0.1.0"

BBOX_PLACEHOLDER = "{{bbox}}"

# Largest allowed box, by how common the searched feature is.
# Rare features need a wider box to find any instance; very common ones need only a small one.
RARITY_MAX_BBOX_KM2 = {
    "rare": 2500.0,
    "medium": 400.0,
    "common": 100.0,
    "very_common": 10.0,
}
DEFAULT_RARITY = "common"
DEFAULT_TIMEOUT_S = 25
MAX_TIMEOUT_S = 90
DEFAULT_MAX_RESULTS = 100
MAX_RESULTS_LIMIT = 1000

EARTH_RADIUS_KM = 6371.0088
# Extra seconds on the HTTP client beyond the server-side timeout, for transfer.
HTTP_MARGIN_S = 10
# Total time for one call across all endpoints. Kept under the MCP client's 60 s
# tool-call timeout, so the tool answers (even with an error) before the client gives up.
TOTAL_BUDGET_S = 50
# Do not start an endpoint attempt with less time than this left in the budget.
MIN_ATTEMPT_S = 5


class OverpassError(ValueError):
    """Raised for invalid input or a failed request. The message is safe to show the model."""


BUSY_STATUS_CODES = {429, 502, 503, 504}


def _error_detail(body: str, limit: int = 500) -> str:
    """Pull the readable error out of an Overpass HTML error page.

    Overpass reports query errors (such as a syntax error) inside the HTML page, after a
    long preamble, so the raw start of the body is not useful.
    """
    text = re.sub(r"<[^>]+>", " ", body)
    text = re.sub(r"\s+", " ", text).strip()
    idx = text.lower().find("error")
    if idx < 0:
        idx = text.lower().find("runtime")
    return (text[idx:] if idx >= 0 else text)[:limit]


def bbox_area_km2(south: float, west: float, north: float, east: float) -> float:
    """Approximate area of a lat/lon box on a sphere, in km²."""
    lon_span = math.radians(east - west)
    return (EARTH_RADIUS_KM ** 2) * lon_span * abs(
        math.sin(math.radians(north)) - math.sin(math.radians(south))
    )


def validate_bbox(south: float, west: float, north: float, east: float, max_km2: float) -> float:
    """Check the box against max_km2 and return its area in km². Raises OverpassError if not allowed."""
    for name, value in (("south", south), ("west", west), ("north", north), ("east", east)):
        if not math.isfinite(value):
            raise OverpassError(f"{name} must be a finite number")
    if not -90 <= south < north <= 90:
        raise OverpassError("need -90 <= south < north <= 90")
    if not -180 <= west < east <= 180:
        raise OverpassError("need -180 <= west < east <= 180 (boxes crossing the antimeridian are not supported)")
    area = bbox_area_km2(south, west, north, east)
    if area > max_km2:
        raise OverpassError(
            f"bounding box is {area:.1f} km², above the maximum of {max_km2:g} km² for this rarity. "
            "Use a smaller box, or a rarer rarity level if the feature is rare."
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


async def _post(endpoint: str, full_query: str, http_timeout: float) -> dict:
    """POST one query to one endpoint and return the parsed JSON. Raises OverpassError on failure."""
    try:
        async with httpx.AsyncClient(
            timeout=httpx.Timeout(http_timeout),
            headers={"User-Agent": USER_AGENT},
        ) as client:
            response = await client.post(endpoint, data={"data": full_query})
    except httpx.TimeoutException as exc:
        raise OverpassError(f"timed out after {http_timeout:.0f} seconds") from exc
    except httpx.HTTPError as exc:
        raise OverpassError(f"request failed: {exc}") from exc

    if response.status_code in BUSY_STATUS_CODES:
        raise OverpassError(f"server busy or overloaded (HTTP {response.status_code})")
    if response.status_code != 200:
        raise OverpassError(f"HTTP {response.status_code}: {_error_detail(response.text)}")

    try:
        return response.json()
    except ValueError as exc:
        raise OverpassError("response is not JSON") from exc


async def _post_with_fallback(full_query: str, timeout: int) -> dict:
    """Try each endpoint in order within TOTAL_BUDGET_S. If all fail, raise one error listing every failure."""
    deadline = time.monotonic() + TOTAL_BUDGET_S
    failures = []
    for endpoint in ENDPOINTS:
        remaining = deadline - time.monotonic()
        if remaining < MIN_ATTEMPT_S:
            failures.append("out of time budget, remaining endpoints skipped")
            break
        http_timeout = min(timeout + HTTP_MARGIN_S, remaining)
        try:
            return await _post(endpoint, full_query, http_timeout)
        except OverpassError as exc:
            failures.append(f"{endpoint}: {exc}")
    raise OverpassError("all Overpass endpoints failed. " + " | ".join(failures))


async def run_query(
    query: str,
    south: float,
    west: float,
    north: float,
    east: float,
    timeout: int = DEFAULT_TIMEOUT_S,
    max_results: int = DEFAULT_MAX_RESULTS,
    rarity: str = DEFAULT_RARITY,
) -> dict:
    if not query or not query.strip():
        raise OverpassError("query must not be empty")
    if rarity not in RARITY_MAX_BBOX_KM2:
        raise OverpassError(f"rarity must be one of: {', '.join(RARITY_MAX_BBOX_KM2)}")
    if not 1 <= timeout <= MAX_TIMEOUT_S:
        raise OverpassError(f"timeout must be between 1 and {MAX_TIMEOUT_S} seconds")
    if not 1 <= max_results <= MAX_RESULTS_LIMIT:
        raise OverpassError(f"max_results must be between 1 and {MAX_RESULTS_LIMIT}")

    area = validate_bbox(south, west, north, east, RARITY_MAX_BBOX_KM2[rarity])
    full_query = build_query(query, south, west, north, east, timeout)

    payload = await _post_with_fallback(full_query, timeout)

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
