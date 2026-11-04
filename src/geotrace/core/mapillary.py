"""Mapillary street-level imagery: search around a point, select references and download thumbnails.

Uses the Mapillary Graph API (`GET https://graph.mapillary.com/images`). Imagery is CC BY-SA 4.0.
"""

import asyncio
import math
import os
import time
from collections import OrderedDict
from collections.abc import Sequence
from dataclasses import dataclass, replace
from typing import Literal

import httpx

GRAPH_IMAGES_URL = "https://graph.mapillary.com/images"
TOKEN_ENV = "MAPILLARY_TOKEN"
FIELDS = (
    "id,captured_at,compass_angle,computed_geometry,geometry,sequence,is_pano,"
    "thumb_256_url,thumb_1024_url"
)
QUERY_LIMIT = 2000
USER_AGENT = "GeoTrace/0.1.0"

SEARCH_CONCURRENCY = 4
THUMBNAIL_CONCURRENCY = 16
REQUEST_TIMEOUT_S = 30.0
THUMBNAIL_TIMEOUT_S = 15.0
MAX_THUMBNAIL_BYTES = 5 * 1024 * 1024

# Search results hold thumbnail URLs, which expire, so cached searches are only reused for a while.
QUERY_CACHE_TTL_S = 10 * 60
QUERY_CACHE_MAX_ENTRIES = 256
# Thumbnails are cached by (image id, size), never by URL. Bounded by total bytes.
THUMBNAIL_CACHE_MAX_BYTES = 128 * 1024 * 1024

EARTH_RADIUS_M = 6_371_008.8
METERS_PER_DEG_LAT = math.pi * EARTH_RADIUS_M / 180

ThumbSize = Literal[256, 1024]


class MapillaryError(RuntimeError):
    """Raised for a failed Mapillary request. The message is safe to show the model."""


class MissingTokenError(MapillaryError):
    """Raised when MAPILLARY_TOKEN is not set."""


@dataclass(frozen=True)
class MapillaryImage:
    id: str
    lat: float
    lon: float
    compass_angle: float | None
    captured_at: int | None          # milliseconds since the Unix epoch
    sequence: str | None
    is_pano: bool
    thumb_256_url: str | None
    thumb_1024_url: str | None
    distance_m: float = 0.0          # distance to the candidate it was selected for
    candidate_index: int = -1        # index of that candidate in the caller's list


def get_token() -> str:
    """Return the Mapillary token from the environment, or raise MissingTokenError with setup steps."""
    token = os.environ.get(TOKEN_ENV, "").strip()
    if not token:
        raise MissingTokenError(
            f"The {TOKEN_ENV} environment variable is not set, so Mapillary street-level photos "
            "cannot be searched. Get a free client access token (it starts with 'MLY|') at "
            "https://www.mapillary.com/dashboard/developers by registering an application, then add "
            f'it to the "env" block of the GeoTrace server in claude_desktop_config.json, for example '
            f'"env": {{"{TOKEN_ENV}": "MLY|..."}}, and restart the client.'
        )
    return token


def haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance between two points, in metres."""
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp, dl = p2 - p1, math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * EARTH_RADIUS_M * math.asin(math.sqrt(min(1.0, a)))


def heading_diff_deg(a: float, b: float) -> float:
    """Smallest angle between two compass headings, in [0, 180]."""
    d = abs(a - b) % 360
    return min(d, 360 - d)


def bbox_around(lat: float, lon: float, radius_m: float) -> tuple[float, float, float, float]:
    """Square box enclosing the circle of radius_m around (lat, lon).

    Returned in Mapillary order: (west, south, east, north). This differs from Overpass,
    which uses (south, west, north, east).
    """
    dlat = radius_m / METERS_PER_DEG_LAT
    cos_lat = max(math.cos(math.radians(lat)), 1e-6)
    dlon = radius_m / (METERS_PER_DEG_LAT * cos_lat)
    south, north = max(lat - dlat, -90.0), min(lat + dlat, 90.0)
    west, east = max(lon - dlon, -180.0), min(lon + dlon, 180.0)
    return west, south, east, north


def format_bbox(bbox: tuple[float, float, float, float]) -> str:
    """The `bbox` query parameter: west,south,east,north."""
    return ",".join(f"{v:.7f}" for v in bbox)


def parse_image(raw: dict) -> MapillaryImage | None:
    """Build a MapillaryImage from one Graph API record. Returns None if it has no usable position.

    computed_geometry (the position after Mapillary's structure-from-motion) is preferred,
    and the raw GPS `geometry` is the fallback. Both are GeoJSON points: [lon, lat].
    """
    coords = None
    for key in ("computed_geometry", "geometry"):
        geom = raw.get(key) or {}
        c = geom.get("coordinates")
        if isinstance(c, (list, tuple)) and len(c) >= 2 and c[0] is not None and c[1] is not None:
            coords = c
            break
    if coords is None or raw.get("id") is None:
        return None
    lon, lat = float(coords[0]), float(coords[1])
    compass = raw.get("compass_angle")
    captured = raw.get("captured_at")
    sequence = raw.get("sequence")
    return MapillaryImage(
        id=str(raw["id"]),
        lat=lat,
        lon=lon,
        compass_angle=float(compass) if compass is not None else None,
        captured_at=int(captured) if captured is not None else None,
        sequence=str(sequence) if sequence else None,
        is_pano=bool(raw.get("is_pano", False)),
        thumb_256_url=raw.get("thumb_256_url"),
        thumb_1024_url=raw.get("thumb_1024_url"),
    )


# ---------------------------------------------------------------------------- search

_query_cache: OrderedDict[tuple[float, float, float], tuple[float, list[MapillaryImage]]] = OrderedDict()


def _cache_key(lat: float, lon: float, radius_m: float) -> tuple[float, float, float]:
    # 5 decimals is about 1 m, so the same candidate sent twice hits the cache.
    return round(lat, 5), round(lon, 5), float(radius_m)


def clear_caches() -> None:
    """Empty the in-memory query and thumbnail caches (used by tests)."""
    global _thumb_cache_bytes
    _query_cache.clear()
    _thumb_cache.clear()
    _thumb_cache_bytes = 0


def _error_message(response: httpx.Response) -> str:
    try:
        return str(response.json().get("error", {}).get("message", ""))[:300]
    except ValueError:
        return response.text[:300]


async def _get_images(client: httpx.AsyncClient, params: dict, token: str) -> list[dict]:
    """One Graph API request, retried once on a timeout or server error."""
    headers = {"Authorization": f"OAuth {token}", "User-Agent": USER_AGENT}
    last_error = ""
    for _attempt in range(2):
        try:
            response = await client.get(GRAPH_IMAGES_URL, params=params, headers=headers)
        except httpx.TimeoutException:
            last_error = f"timed out after {REQUEST_TIMEOUT_S:.0f} seconds"
            continue
        except httpx.HTTPError as exc:
            raise MapillaryError(f"Mapillary request failed: {exc}") from exc
        if response.status_code in (401, 403):
            raise MapillaryError(
                f"Mapillary rejected the token (HTTP {response.status_code}: {_error_message(response)}). "
                f"Check that {TOKEN_ENV} is a valid client access token (it starts with 'MLY|')."
            )
        if response.status_code == 429:
            raise MapillaryError("Mapillary rate limit reached (HTTP 429). Wait a minute and try again.")
        if response.status_code >= 500:
            last_error = f"HTTP {response.status_code}"
            continue
        if response.status_code != 200:
            raise MapillaryError(f"Mapillary HTTP {response.status_code}: {_error_message(response)}")
        try:
            data = response.json().get("data", [])
        except ValueError as exc:
            raise MapillaryError("Mapillary response is not JSON") from exc
        return data if isinstance(data, list) else []
    raise MapillaryError(f"Mapillary request failed: {last_error}")


async def mapillary_images(
    lat: float,
    lon: float,
    radius_m: float,
    token: str,
    client: httpx.AsyncClient | None = None,
) -> list[MapillaryImage]:
    """All Mapillary images in the square box around (lat, lon). Cached by (rounded point, radius)."""
    key = _cache_key(lat, lon, radius_m)
    hit = _query_cache.get(key)
    if hit is not None and time.monotonic() - hit[0] < QUERY_CACHE_TTL_S:
        _query_cache.move_to_end(key)
        return hit[1]

    params = {
        "bbox": format_bbox(bbox_around(lat, lon, radius_m)),
        "fields": FIELDS,
        "limit": QUERY_LIMIT,
    }
    if client is None:
        async with httpx.AsyncClient(timeout=REQUEST_TIMEOUT_S) as own:
            raw = await _get_images(own, params, token)
    else:
        raw = await _get_images(client, params, token)

    images = [img for img in (parse_image(r) for r in raw) if img is not None]
    _query_cache[key] = (time.monotonic(), images)
    while len(_query_cache) > QUERY_CACHE_MAX_ENTRIES:
        _query_cache.popitem(last=False)
    return images


def select_images(
    images: Sequence[MapillaryImage],
    center: tuple[float, float],
    radius_m: float,
    min_spacing_m: float = 12,
    min_heading_diff: float = 45,
    max_images: int = 100,
) -> list[MapillaryImage]:
    """Pick a diverse set of reference images around center (lat, lon), nearest first.

    - Images outside the circle are dropped (the search box is a square).
    - Within one sequence, an image is dropped if an already selected image of that sequence is
      closer than min_spacing_m AND faces a similar direction (compass difference below
      min_heading_diff). Images facing a different direction are kept. An image with an unknown
      heading counts as facing the same direction.
    - Panoramas are skipped.
    Images are considered nearest first, so the closest of a near-duplicate group is the one kept.
    """
    lat0, lon0 = center
    in_circle: list[MapillaryImage] = []
    for img in images:
        # TODO: support panoramas (is_pano) by cutting them into perspective crops, as the
        # MegaLoc authors recommend. Skipped in this first version.
        if img.is_pano:
            continue
        d = haversine_m(lat0, lon0, img.lat, img.lon)
        if d <= radius_m:
            in_circle.append(replace(img, distance_m=d))
    in_circle.sort(key=lambda i: (i.distance_m, i.id))

    selected: list[MapillaryImage] = []
    by_sequence: dict[str, list[MapillaryImage]] = {}
    for img in in_circle:
        if len(selected) >= max_images:
            break
        if img.sequence is not None:
            kept = by_sequence.setdefault(img.sequence, [])
            if any(_near_duplicate(img, k, min_spacing_m, min_heading_diff) for k in kept):
                continue
            kept.append(img)
        selected.append(img)
    return selected


def _near_duplicate(a: MapillaryImage, b: MapillaryImage, min_spacing_m: float, min_heading_diff: float) -> bool:
    if haversine_m(a.lat, a.lon, b.lat, b.lon) >= min_spacing_m:
        return False
    if a.compass_angle is None or b.compass_angle is None:
        return True
    return heading_diff_deg(a.compass_angle, b.compass_angle) < min_heading_diff


async def candidate_image_ids(
    candidates: Sequence[tuple[float, float]],
    token: str,
    radius_m: float,
    per_candidate: int,
    *,
    index_offset: int = 0,
    seen_ids: set[str] | None = None,
    client: httpx.AsyncClient | None = None,
) -> tuple[list[MapillaryImage], list[int]]:
    """Search and select reference images for each (lat, lon) candidate.

    Requests run concurrently (at most SEARCH_CONCURRENCY at a time). Candidate i is reported with
    index index_offset + i. An image near several candidates is kept once, for the first (most
    likely) of them; pass the same seen_ids set across calls to deduplicate across batches.

    Returns (selected, indices_without_coverage): the selected images, each with its
    candidate_index set, and the indices of candidates with no usable image in the circle.
    """
    seen = seen_ids if seen_ids is not None else set()
    semaphore = asyncio.Semaphore(SEARCH_CONCURRENCY)

    async def search(c: httpx.AsyncClient, lat: float, lon: float) -> list[MapillaryImage]:
        async with semaphore:
            return await mapillary_images(lat, lon, radius_m, token, client=c)

    async def search_all(c: httpx.AsyncClient) -> list[list[MapillaryImage]]:
        return list(await asyncio.gather(*(search(c, lat, lon) for lat, lon in candidates)))

    if client is None:
        async with httpx.AsyncClient(timeout=REQUEST_TIMEOUT_S) as own:
            results = await search_all(own)
    else:
        results = await search_all(client)

    selected: list[MapillaryImage] = []
    without_coverage: list[int] = []
    for i, ((lat, lon), found) in enumerate(zip(candidates, results)):
        index = index_offset + i
        chosen = select_images(found, (lat, lon), radius_m, max_images=per_candidate)
        if not chosen:
            without_coverage.append(index)
            continue
        for img in chosen:
            if img.id in seen:
                continue
            seen.add(img.id)
            selected.append(replace(img, candidate_index=index))
    return selected, without_coverage


# ---------------------------------------------------------------------------- thumbnails

_thumb_cache: OrderedDict[tuple[str, int], bytes] = OrderedDict()
_thumb_cache_bytes = 0


def _cache_thumb(key: tuple[str, int], data: bytes) -> None:
    global _thumb_cache_bytes
    if key in _thumb_cache:
        return
    _thumb_cache[key] = data
    _thumb_cache_bytes += len(data)
    while _thumb_cache_bytes > THUMBNAIL_CACHE_MAX_BYTES and _thumb_cache:
        _, old = _thumb_cache.popitem(last=False)
        _thumb_cache_bytes -= len(old)


async def download_thumbnails(
    images: Sequence[MapillaryImage],
    size: ThumbSize,
    client: httpx.AsyncClient | None = None,
) -> dict[str, bytes]:
    """Download the size-px thumbnail of each image, at most THUMBNAIL_CONCURRENCY at a time.

    Thumbnail URLs expire, so this should be called soon after the search. Results are cached by
    (image id, size). Images whose thumbnail is missing or fails to download are left out.
    """
    out: dict[str, bytes] = {}
    todo: list[tuple[str, str]] = []
    for img in images:
        cached = _thumb_cache.get((img.id, size))
        if cached is not None:
            _thumb_cache.move_to_end((img.id, size))
            out[img.id] = cached
            continue
        url = img.thumb_256_url if size == 256 else img.thumb_1024_url
        if url:
            todo.append((img.id, url))
    if not todo:
        return out

    semaphore = asyncio.Semaphore(THUMBNAIL_CONCURRENCY)

    async def fetch(c: httpx.AsyncClient, image_id: str, url: str) -> None:
        async with semaphore:
            try:
                response = await c.get(url, headers={"User-Agent": USER_AGENT})
            except httpx.HTTPError:
                return
        if response.status_code != 200 or not response.content or len(response.content) > MAX_THUMBNAIL_BYTES:
            return
        _cache_thumb((image_id, size), response.content)
        out[image_id] = response.content

    async def fetch_all(c: httpx.AsyncClient) -> None:
        await asyncio.gather(*(fetch(c, image_id, url) for image_id, url in todo))

    if client is None:
        async with httpx.AsyncClient(timeout=THUMBNAIL_TIMEOUT_S, follow_redirects=True) as own:
            await fetch_all(own)
    else:
        await fetch_all(client)
    return out
