"""Reverse search tests. No real network: HTTP goes through httpx.MockTransport and the face
detector is replaced by a stub."""
import io
import json
from pathlib import Path

import httpx
import pytest
from PIL import Image

from geotrace.core import reverse_search as rs

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    """Overrides conftest's guard: still refuses any real request, but lets clients built on
    httpx.MockTransport through, so the backends' HTTP handling can be tested."""
    send = httpx.AsyncClient.send

    async def guarded(self, request, **kwargs):
        if isinstance(self._transport, httpx.MockTransport):
            return await send(self, request, **kwargs)
        raise RuntimeError("network access is disabled in tests")

    monkeypatch.setattr(httpx.AsyncClient, "send", guarded)


def test_guard_still_blocks_real_transports():
    async def go():
        async with httpx.AsyncClient() as client:
            await client.get("https://example.com")
    import asyncio
    with pytest.raises(RuntimeError, match="network access is disabled"):
        asyncio.run(go())


def fixture(name: str) -> dict:
    return json.loads((FIXTURES / name).read_text())


def make_jpeg(w: int = 2000, h: int = 2000, noisy: bool = False) -> bytes:
    if noisy:
        import os
        img = Image.frombytes("RGB", (w, h), os.urandom(w * h * 3))
    else:
        img = Image.new("RGB", (w, h), (90, 120, 200))
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=95)
    return buf.getvalue()


def no_face(_img: Image.Image) -> float:
    return 0.0


class FakeBackend:
    def __init__(self, name: str = "serpapi", result: rs.ReverseSearchResult | None = None):
        self.name = name
        self.calls: list[bytes] = []
        self.result = result or rs.ReverseSearchResult(backend=name, best_guess=["x"])

    async def search(self, jpeg: bytes) -> rs.ReverseSearchResult:
        self.calls.append(jpeg)
        return self.result


# --- normalisation -----------------------------------------------------------------------------

def test_normalize_serpapi():
    r = rs.normalize_serpapi(fixture("serpapi_lens_all.json"))
    assert r.backend == "serpapi"
    assert r.best_guess == ["Torre del Reloj", "Clock tower"]
    assert r.pages == [{"url": "https://es.wikipedia.org/wiki/Torre_del_Reloj",
                        "title": "Torre del Reloj - Wikipedia", "source": "Wikipedia"}]
    assert len(r.visual_matches) == rs.MAX_ITEMS
    assert r.visual_matches[0] == {"url": "https://example0.org/tower", "title": "Clock tower view 1",
                                   "source": "Example 0", "thumbnail": "https://encrypted-tbn2.gstatic.com/images?q=tbn:0"}
    assert r.exact_matches == []
    assert r.total_results == 19          # 1 page + 18 visual matches
    assert r.truncated is True


def test_normalize_serpapi_exact_matches():
    r = rs.normalize_serpapi(fixture("serpapi_lens_exact.json"))
    assert r.exact_matches == ["https://serpapi.com/searches/x/images/exact1.jpeg"]
    assert r.truncated is False


def test_normalize_serpapi_empty():
    r = rs.normalize_serpapi({"search_metadata": {"status": "Success"}})
    assert r.total_results == 0 and r.visual_matches == [] and not r.truncated


def test_normalize_vision():
    r = rs.normalize_vision(fixture("vision_web_detection.json"))
    assert r.backend == "cloud_vision"
    assert r.best_guess == ["torre del reloj", "Torre del Reloj", "Clock tower"]
    assert r.pages == [{"url": "https://es.wikipedia.org/wiki/Torre_del_Reloj",
                        "title": "Torre del Reloj - Wikipedia", "source": "es.wikipedia.org"}]
    assert r.exact_matches == ["https://upload.wikimedia.org/tower.jpg"]
    assert [v["title"] for v in r.visual_matches] == ["partial match", "visually similar", "visually similar"]
    assert r.visual_matches[1]["source"] == "similar.example.net"
    assert r.total_results == 5 and not r.truncated


# --- backend selection -------------------------------------------------------------------------

def test_serpapi_is_default_when_configured():
    backend, note = rs.select_backend("serpapi", {rs.SERPAPI_ENV: "k", rs.VISION_ENV: "v"})
    assert backend.name == "serpapi" and note is None


def test_fallback_to_cloud_vision_with_note():
    backend, note = rs.select_backend("serpapi", {rs.VISION_ENV: "v"})
    assert backend.name == "cloud_vision"
    assert note and rs.SERPAPI_ENV in note and "Cloud Vision" in note


def test_explicit_cloud_vision():
    backend, note = rs.select_backend("cloud_vision", {rs.SERPAPI_ENV: "k", rs.VISION_ENV: "v"})
    assert backend.name == "cloud_vision" and note is None


def test_cloud_vision_requested_but_only_serpapi_configured():
    with pytest.raises(rs.NotConfiguredError, match="backend='serpapi'"):
        rs.select_backend("cloud_vision", {rs.SERPAPI_ENV: "k"})


@pytest.mark.parametrize("requested", ["serpapi", "cloud_vision"])
def test_nothing_configured_lists_env_vars(requested):
    with pytest.raises(rs.NotConfiguredError) as exc:
        rs.select_backend(requested, {rs.SERPAPI_ENV: "  "})
    assert rs.SERPAPI_ENV in str(exc.value) and rs.VISION_ENV in str(exc.value)


# --- SerpApi backend over a mocked transport ---------------------------------------------------

def serpapi_transport(seen: list[httpx.Request], search_response: httpx.Response | None = None,
                      upload_response: httpx.Response | None = None) -> httpx.MockTransport:
    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        if request.url.path == "/image":
            return upload_response or httpx.Response(
                200, json={"message": "Image uploaded successfully.", "image_id": "IMG123"})
        if request.url.path == "/search":
            return search_response or httpx.Response(200, json=fixture("serpapi_lens_all.json"))
        return httpx.Response(404)
    return httpx.MockTransport(handler)


async def test_serpapi_uploads_then_searches_by_image_id():
    seen: list[httpx.Request] = []
    backend = rs.SerpApiBackend("KEY", serpapi_transport(seen))
    result = await backend.search(make_jpeg())
    upload, search = seen
    assert upload.method == "POST" and str(upload.url) == rs.SERPAPI_UPLOAD_URL
    assert b'name="image"' in upload.content and b'name="api_key"' in upload.content
    assert search.url.params["engine"] == "google_lens"
    assert search.url.params["image_id"] == "IMG123"
    assert "url" not in search.url.params
    assert result.best_guess[0] == "Torre del Reloj"


async def test_serpapi_upload_is_shrunk_under_limit():
    seen: list[httpx.Request] = []
    big = make_jpeg(1568, 1568, noisy=True)
    assert len(big) > rs.SERPAPI_MAX_UPLOAD_BYTES
    await rs.SerpApiBackend("KEY", serpapi_transport(seen)).search(big)
    assert len(seen[0].content) < rs.SERPAPI_MAX_UPLOAD_BYTES + 1000   # multipart overhead only


def test_fit_jpeg_keeps_small_images_untouched():
    small = make_jpeg(300, 300)
    assert rs.fit_jpeg(small, rs.SERPAPI_MAX_UPLOAD_BYTES) is small


@pytest.mark.parametrize("status,body,match", [
    (401, {"error": "Invalid API key. Your API key should be here: https://serpapi.com/manage-api-key"}, "SERPAPI_API_KEY"),
    (429, {"error": "Your account has run out of searches."}, "run out of searches"),
    (429, {"error": "Hourly throughput exceeded"}, "rate limit"),
    (503, {"error": "We couldn't get valid results for this search. Please try again later."}, "server error"),
])
async def test_serpapi_http_errors_are_actionable(status, body, match):
    backend = rs.SerpApiBackend("KEY", serpapi_transport([], httpx.Response(status, json=body)))
    with pytest.raises(rs.BackendError, match=match):
        await backend.search(make_jpeg(300, 300))


async def test_serpapi_upload_error():
    bad = httpx.Response(200, json={"error": "Invalid image format. Supported format: jpg, jpeg, png, webp"})
    backend = rs.SerpApiBackend("KEY", serpapi_transport([], upload_response=bad))
    with pytest.raises(rs.BackendError, match="Invalid image format"):
        await backend.search(make_jpeg(300, 300))


async def test_serpapi_no_results_is_empty_not_error():
    empty = httpx.Response(200, json={"search_metadata": {"status": "Success"},
                                      "error": "Google Lens hasn't returned any results for this query."})
    result = await rs.SerpApiBackend("KEY", serpapi_transport([], empty)).search(make_jpeg(300, 300))
    assert result.total_results == 0


async def test_serpapi_timeout():
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("slow", request=request)
    backend = rs.SerpApiBackend("KEY", httpx.MockTransport(handler))
    with pytest.raises(rs.BackendError, match="timed out"):
        await backend.search(make_jpeg(300, 300))


# --- Cloud Vision backend over a mocked transport ----------------------------------------------

async def test_cloud_vision_request_and_mapping():
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, json=fixture("vision_web_detection.json"))

    result = await rs.CloudVisionBackend("VKEY", httpx.MockTransport(handler)).search(make_jpeg(300, 300))
    (req,) = seen
    assert req.url.params["key"] == "VKEY"
    body = json.loads(req.content)["requests"][0]
    assert body["features"] == [{"type": "WEB_DETECTION", "maxResults": 20}]
    assert body["image"]["content"]
    assert result.exact_matches == ["https://upload.wikimedia.org/tower.jpg"]


@pytest.mark.parametrize("status,message,match", [
    (400, "API key not valid. Please pass a valid API key.", "GOOGLE_VISION_API_KEY"),
    (403, "Cloud Vision API has not been used in project 1 before or it is disabled.", "enabled"),
    (429, "Quota exceeded", "quota"),
])
async def test_cloud_vision_errors(status, message, match):
    transport = httpx.MockTransport(lambda r: httpx.Response(status, json={"error": {"code": status, "message": message}}))
    with pytest.raises(rs.BackendError, match=match):
        await rs.CloudVisionBackend("VKEY", transport).search(make_jpeg(300, 300))


# --- guardrails and caching --------------------------------------------------------------------

async def test_region_too_small_in_original_pixels():
    data = make_jpeg(1000, 1000)                       # 0.1 of 1000 px = 100 px < 200
    with pytest.raises(rs.CropRejectedError, match="at least 200 px"):
        await rs.search_region(data, (0, 0, 0.1, 0.5), FakeBackend(), no_face, cache=rs.ResultCache())


async def test_region_exactly_200px_is_accepted():
    data = make_jpeg(1000, 1000)
    backend = FakeBackend()
    await rs.search_region(data, (0, 0, 0.2, 0.2), backend, no_face, cache=rs.ResultCache())
    assert len(backend.calls) == 1


async def test_face_guard_refuses_and_does_not_search():
    backend = FakeBackend()
    with pytest.raises(rs.CropRejectedError, match="people"):
        await rs.search_region(make_jpeg(), (0, 0, 0.5, 0.5), backend, lambda _img: 0.25, cache=rs.ResultCache())
    assert backend.calls == []


async def test_small_face_is_allowed():
    backend = FakeBackend()
    await rs.search_region(make_jpeg(), (0, 0, 0.5, 0.5), backend, lambda _img: 0.05, cache=rs.ResultCache())
    assert len(backend.calls) == 1


async def test_face_detector_sees_original_resolution_crop():
    sizes = []

    def spy(img: Image.Image) -> float:
        sizes.append(img.size)
        return 0.0

    await rs.search_region(make_jpeg(4000, 3000), (0.5, 0.25, 0.75, 0.5), FakeBackend(), spy, cache=rs.ResultCache())
    assert sizes == [(1000, 750)]


async def test_same_crop_is_searched_once():
    cache = rs.ResultCache()
    backend = FakeBackend()
    data = make_jpeg()
    first, info1 = await rs.search_region(data, (0, 0, 0.5, 0.5), backend, no_face, cache=cache)
    second, info2 = await rs.search_region(data, (0, 0, 0.5, 0.5), backend, no_face, cache=cache)
    assert len(backend.calls) == 1
    assert info1["cached"] is False and info2["cached"] is True
    assert first == second


async def test_cache_is_per_backend_and_per_crop():
    cache = rs.ResultCache()
    serp, vision = FakeBackend("serpapi"), FakeBackend("cloud_vision")
    data = make_jpeg()
    await rs.search_region(data, (0, 0, 0.5, 0.5), serp, no_face, cache=cache)
    await rs.search_region(data, (0, 0, 0.5, 0.5), vision, no_face, cache=cache)
    await rs.search_region(make_jpeg(noisy=True), (0, 0, 0.5, 0.5), serp, no_face, cache=cache)
    assert len(serp.calls) == 2 and len(vision.calls) == 1


async def test_failed_search_is_not_cached():
    cache = rs.ResultCache()

    class Failing(FakeBackend):
        async def search(self, jpeg: bytes) -> rs.ReverseSearchResult:
            self.calls.append(jpeg)
            raise rs.BackendError("boom")

    backend = Failing()
    for _ in range(2):
        with pytest.raises(rs.BackendError):
            await rs.search_region(make_jpeg(), (0, 0, 0.5, 0.5), backend, no_face, cache=cache)
    assert len(backend.calls) == 2


async def test_fallback_note_is_attached_and_cache_stays_clean():
    cache = rs.ResultCache()
    backend = FakeBackend("cloud_vision")
    result, _ = await rs.search_region(make_jpeg(), (0, 0, 0.5, 0.5), backend, no_face, note="fell back", cache=cache)
    again, _ = await rs.search_region(make_jpeg(), (0, 0, 0.5, 0.5), backend, no_face, cache=cache)
    assert result.note == "fell back" and again.note is None


def test_cache_evicts_oldest():
    cache = rs.ResultCache(max_entries=2)
    for i in range(3):
        cache.put(("serpapi", str(i)), rs.ReverseSearchResult(backend="serpapi"))
    assert cache.get(("serpapi", "0")) is None and cache.get(("serpapi", "2")) is not None
