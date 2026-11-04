"""Tests for core/mapillary.py. All HTTP goes through httpx.MockTransport: no real requests."""

import itertools

import httpx
import pytest

from geotrace.core import mapillary
from geotrace.core.mapillary import MapillaryImage, haversine_m, select_images

LAT, LON = 40.4168, -3.7038          # Madrid, Puerta del Sol
M_PER_DEG_LAT = mapillary.METERS_PER_DEG_LAT


@pytest.fixture(autouse=True)
def _clean_caches():
    mapillary.clear_caches()
    yield
    mapillary.clear_caches()


def north_of(meters: float, east_m: float = 0.0) -> tuple[float, float]:
    """A point `meters` north and `east_m` east of (LAT, LON)."""
    import math

    return (
        LAT + meters / M_PER_DEG_LAT,
        LON + east_m / (M_PER_DEG_LAT * math.cos(math.radians(LAT))),
    )


def img(image_id: str, north_m: float, east_m: float = 0.0, heading: float | None = 0.0,
        sequence: str | None = "s1", pano: bool = False) -> MapillaryImage:
    lat, lon = north_of(north_m, east_m)
    return MapillaryImage(
        id=image_id, lat=lat, lon=lon, compass_angle=heading, captured_at=1_600_000_000_000,
        sequence=sequence, is_pano=pano, thumb_256_url=f"https://t/{image_id}/256",
        thumb_1024_url=f"https://t/{image_id}/1024",
    )


def raw(image_id: str, lat: float, lon: float, **extra) -> dict:
    return {
        "id": image_id,
        "computed_geometry": {"type": "Point", "coordinates": [lon, lat]},
        "compass_angle": 90.0,
        "captured_at": 1_600_000_000_000,
        "sequence": "seq",
        "is_pano": False,
        "thumb_256_url": f"https://t/{image_id}/256",
        "thumb_1024_url": f"https://t/{image_id}/1024",
        **extra,
    }


# ---------------------------------------------------------------------------- select_images

def test_select_images_drops_corners_of_the_square_box():
    # 90 m north and 90 m east is inside the 100 m box but about 127 m away: outside the circle.
    inside = img("inside", 90, sequence=None)
    corner = img("corner", 90, 90, sequence=None)
    chosen = select_images([inside, corner], (LAT, LON), radius_m=100)
    assert [i.id for i in chosen] == ["inside"]
    assert haversine_m(LAT, LON, corner.lat, corner.lon) > 100


def test_select_images_spacing_within_a_sequence():
    images = [img("a", 0), img("b", 5), img("c", 20)]          # b is 5 m from a, same heading
    chosen = select_images(images, (LAT, LON), radius_m=100, min_spacing_m=12)
    assert [i.id for i in chosen] == ["a", "c"]


def test_select_images_spacing_does_not_apply_across_sequences():
    images = [img("a", 0, sequence="s1"), img("b", 5, sequence="s2")]
    chosen = select_images(images, (LAT, LON), radius_m=100)
    assert {i.id for i in chosen} == {"a", "b"}


def test_select_images_keeps_close_images_facing_another_way():
    images = [
        img("north", 0, heading=0),
        img("east", 2, heading=90),          # 90° apart: kept
        img("north2", 3, heading=30),        # 30° from "north": dropped
        img("wrap", 4, heading=350),         # 10° from "north" across 0°: dropped
    ]
    chosen = select_images(images, (LAT, LON), radius_m=100, min_heading_diff=45)
    assert [i.id for i in chosen] == ["north", "east"]


def test_select_images_unknown_heading_counts_as_similar():
    images = [img("a", 0, heading=None), img("b", 3, heading=180)]
    chosen = select_images(images, (LAT, LON), radius_m=100)
    assert [i.id for i in chosen] == ["a"]


def test_select_images_sorted_by_distance_and_capped():
    images = [img(f"i{d}", d, sequence=None) for d in (80, 10, 50, 30, 0)]
    chosen = select_images(images, (LAT, LON), radius_m=100, max_images=3)
    assert [i.id for i in chosen] == ["i0", "i10", "i30"]
    assert all(a.distance_m <= b.distance_m for a, b in itertools.pairwise(chosen))


def test_select_images_skips_panoramas():
    chosen = select_images([img("p", 0, pano=True), img("q", 30)], (LAT, LON), radius_m=100)
    assert [i.id for i in chosen] == ["q"]


# ---------------------------------------------------------------------------- bbox

def test_bbox_around_is_west_south_east_north():
    west, south, east, north = mapillary.bbox_around(LAT, LON, 150)
    assert west < LON < east
    assert south < LAT < north
    assert haversine_m(south, LON, north, LON) == pytest.approx(300, rel=0.01)
    assert haversine_m(LAT, west, LAT, east) == pytest.approx(300, rel=0.01)


@pytest.mark.anyio
async def test_mapillary_request_sends_bbox_in_west_south_east_north_order():
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(200, json={"data": [raw("1", LAT, LON)]})

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        found = await mapillary.mapillary_images(LAT, LON, 150, "MLY|test", client=client)

    assert [i.id for i in found] == ["1"]
    params = requests[0].url.params
    west, south, east, north = (float(v) for v in params["bbox"].split(","))
    assert west < east and south < north
    assert west < LON < east and south < LAT < north          # lon first, then lat
    assert params["limit"] == "2000"
    assert set(params["fields"].split(",")) == {
        "id", "captured_at", "compass_angle", "computed_geometry", "geometry", "sequence",
        "is_pano", "thumb_256_url", "thumb_1024_url",
    }
    assert requests[0].headers["Authorization"] == "OAuth MLY|test"
    assert "access_token" not in params


def test_parse_image_reads_lon_lat_and_falls_back_to_geometry():
    computed = mapillary.parse_image(raw("1", 10.0, 20.0))
    assert computed is not None and (computed.lat, computed.lon) == (10.0, 20.0)

    fallback = mapillary.parse_image(
        raw("2", 0, 0, computed_geometry=None, geometry={"type": "Point", "coordinates": [20.0, 10.0]})
    )
    assert fallback is not None and (fallback.lat, fallback.lon) == (10.0, 20.0)
    assert mapillary.parse_image({"id": "3"}) is None


# ---------------------------------------------------------------------------- candidates

def _point_handler(images_by_candidate: dict[tuple[float, float], list[dict]]):
    """A mock Graph API that returns the images of the candidate at the centre of the bbox."""

    def handler(request: httpx.Request) -> httpx.Response:
        west, south, east, north = (float(v) for v in request.url.params["bbox"].split(","))
        center = ((south + north) / 2, (west + east) / 2)
        point = min(images_by_candidate, key=lambda p: haversine_m(*p, *center))
        return httpx.Response(200, json={"data": images_by_candidate[point]})

    return handler


@pytest.mark.anyio
async def test_candidate_image_ids_deduplicates_and_reports_coverage():
    a = (LAT, LON)
    b = north_of(40)                        # 40 m from a: shares photos with it
    c = north_of(5000)                      # far away, no photos
    shared = raw("shared", *north_of(20))
    only_a = raw("only_a", *north_of(-20))
    only_b = raw("only_b", *north_of(60))
    handler = _point_handler({a: [shared, only_a], b: [shared, only_b], c: []})

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        selected, no_cov = await mapillary.candidate_image_ids(
            [a, b, c], "MLY|test", radius_m=100, per_candidate=50, index_offset=10, client=client,
        )

    by_id = {i.id: i.candidate_index for i in selected}
    assert by_id == {"shared": 10, "only_a": 10, "only_b": 11}  # shared goes to the first candidate
    assert len(selected) == 3
    assert no_cov == [12]


@pytest.mark.anyio
async def test_candidate_image_ids_deduplicates_across_calls_with_seen_ids():
    a = (LAT, LON)
    handler = _point_handler({a: [raw("x", LAT, LON)]})
    seen: set[str] = set()
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        first, _ = await mapillary.candidate_image_ids([a], "t", 100, 50, seen_ids=seen, client=client)
        second, no_cov = await mapillary.candidate_image_ids([a], "t", 100, 50, seen_ids=seen, client=client)
    assert [i.id for i in first] == ["x"]
    assert second == [] and no_cov == []    # covered, but its photo was already used


@pytest.mark.anyio
async def test_queries_are_cached_by_point_and_radius():
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(200, json={"data": []})

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        await mapillary.mapillary_images(LAT, LON, 150, "t", client=client)
        await mapillary.mapillary_images(LAT + 1e-7, LON, 150, "t", client=client)   # same rounded point
        await mapillary.mapillary_images(LAT, LON, 300, "t", client=client)          # other radius
    assert calls == 2


@pytest.mark.anyio
async def test_rejected_token_raises_actionable_error():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(401, json={"error": {"message": "Invalid OAuth access token"}})

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        with pytest.raises(mapillary.MapillaryError, match="MAPILLARY_TOKEN"):
            await mapillary.mapillary_images(LAT, LON, 150, "bad", client=client)


def test_get_token_missing(monkeypatch):
    monkeypatch.delenv("MAPILLARY_TOKEN", raising=False)
    with pytest.raises(mapillary.MissingTokenError, match="claude_desktop_config.json"):
        mapillary.get_token()


@pytest.mark.anyio
async def test_thumbnails_are_cached_by_id_not_url():
    calls: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(str(request.url))
        return httpx.Response(200, content=b"jpeg-bytes")

    first = img("a", 0)
    # Same image id, new (refreshed) URL: must come from the cache.
    second = MapillaryImage(**{**first.__dict__, "thumb_256_url": "https://t/a/256?signature=new"})
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        got1 = await mapillary.download_thumbnails([first], 256, client=client)
        got2 = await mapillary.download_thumbnails([second], 256, client=client)
    assert got1 == got2 == {"a": b"jpeg-bytes"}
    assert len(calls) == 1
