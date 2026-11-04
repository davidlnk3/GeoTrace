"""Reverse image search of a region of an image, with interchangeable backends.

Backends:
- serpapi (default): SerpApi's Google Lens engine. The crop is uploaded to SerpApi's own
  image endpoint (POST https://serpapi.com/image, max 500 KB, the returned image_id expires
  after 10 minutes) and then searched with engine=google_lens&image_id=...
  No third-party image host is ever used.
- cloud_vision: Google Cloud Vision Web Detection, with the crop sent inline as base64.

Every backend returns the same ReverseSearchResult schema.
"""
import asyncio
import base64
import hashlib
import io
from collections import OrderedDict
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass, field
from typing import Any, Literal, Protocol
from urllib.parse import urlparse

import httpx
from PIL import Image

from geotrace.core import zoom

BackendName = Literal["serpapi", "cloud_vision"]

MAX_ITEMS = 15                    # per list returned to the model
MIN_CROP_PX = 200                 # minimum crop side in the ORIGINAL image
FACE_MAX_FRACTION = 0.10          # refuse if one detected face covers more than this share of the crop

SERPAPI_ENV = "SERPAPI_API_KEY"
VISION_ENV = "GOOGLE_VISION_API_KEY"

SERPAPI_UPLOAD_URL = "https://serpapi.com/image"
SERPAPI_SEARCH_URL = "https://serpapi.com/search"
SERPAPI_MAX_UPLOAD_BYTES = 500_000    # documented limit is "500 KB"; use the stricter decimal reading
VISION_URL = "https://vision.googleapis.com/v1/images:annotate"
VISION_MAX_RESULTS = 20

UPLOAD_TIMEOUT_S = 20
SEARCH_TIMEOUT_S = 45

CONFIG_HELP = (
    f"Option 1 (default, SerpApi Google Lens): set {SERPAPI_ENV}. "
    f"Option 2 (Google Cloud Vision Web Detection): set {VISION_ENV}. "
    "Set them in the \"env\" block of the GeoTrace entry in claude_desktop_config.json and restart the client."
)


# --- errors ------------------------------------------------------------------------------------

class ReverseSearchError(Exception):
    """Any failure with a message the model can act on."""


class NotConfiguredError(ReverseSearchError):
    pass


class CropRejectedError(ReverseSearchError):
    pass


class BackendError(ReverseSearchError):
    pass


# --- result schema -----------------------------------------------------------------------------

@dataclass
class ReverseSearchResult:
    backend: str                                              # "serpapi" | "cloud_vision"
    best_guess: list[str] = field(default_factory=list)       # labels / entity names, if any
    pages: list[dict] = field(default_factory=list)           # {"url", "title", "source"}
    exact_matches: list[str] = field(default_factory=list)    # image URLs
    visual_matches: list[dict] = field(default_factory=list)  # {"url", "title", "source", "thumbnail"}
    total_results: int = 0                                    # items found before truncation
    truncated: bool = False
    note: str | None = None                                   # e.g. why a fallback backend was used

    def as_dict(self) -> dict:
        return asdict(self)


def _finish(backend: str, best_guess: list[str], pages: list[dict], exact: list[str], visual: list[dict]) -> ReverseSearchResult:
    lists = (best_guess, pages, exact, visual)
    total = len(pages) + len(exact) + len(visual)
    return ReverseSearchResult(
        backend=backend,
        best_guess=best_guess[:MAX_ITEMS],
        pages=pages[:MAX_ITEMS],
        exact_matches=exact[:MAX_ITEMS],
        visual_matches=visual[:MAX_ITEMS],
        total_results=total,
        truncated=any(len(x) > MAX_ITEMS for x in lists),
    )


def _dedup(items: list[str]) -> list[str]:
    return list(dict.fromkeys(i for i in items if i))


def _host(url: str | None) -> str:
    return urlparse(url or "").hostname or ""


# --- response normalisation ----------------------------------------------------------------------

def normalize_serpapi(data: dict) -> ReverseSearchResult:
    """Maps a SerpApi google_lens response.

    best_guess: related_content[].query (SerpApi's Lens output has no "best guess" field).
    pages: organic_results. visual_matches: visual_matches. exact_matches: the exact_matches
    array, which SerpApi documents for type=exact_matches; it is read when present.
    """
    best = _dedup([str(r.get("query", "")) for r in data.get("related_content") or []])
    pages = [
        {"url": r.get("link"), "title": r.get("title"), "source": r.get("source") or _host(r.get("link"))}
        for r in data.get("organic_results") or [] if r.get("link")
    ]
    visual = [
        {"url": r.get("link"), "title": r.get("title"), "source": r.get("source") or _host(r.get("link")),
         "thumbnail": r.get("thumbnail")}
        for r in data.get("visual_matches") or [] if r.get("link")
    ]
    exact = _dedup([r.get("thumbnail") or r.get("link") or "" for r in data.get("exact_matches") or []])
    return _finish("serpapi", best, pages, exact, visual)


def normalize_vision(data: dict) -> ReverseSearchResult:
    """Maps a Cloud Vision images:annotate response with WEB_DETECTION.

    exact_matches: fullMatchingImages. visual_matches: partialMatchingImages (crops or edits of
    the same picture) followed by visuallySimilarImages. best_guess: bestGuessLabels, then web
    entity descriptions.
    """
    responses = data.get("responses") or [{}]
    web = responses[0].get("webDetection") or {}
    best = _dedup(
        [lbl.get("label", "") for lbl in web.get("bestGuessLabels") or []]
        + [ent.get("description", "") for ent in web.get("webEntities") or []]
    )
    pages = [
        {"url": p.get("url"), "title": _strip_tags(p.get("pageTitle") or ""), "source": _host(p.get("url"))}
        for p in web.get("pagesWithMatchingImages") or [] if p.get("url")
    ]
    exact = _dedup([i.get("url", "") for i in web.get("fullMatchingImages") or []])
    visual = [
        {"url": i["url"], "title": kind, "source": _host(i["url"]), "thumbnail": i["url"]}
        for kind, key in (("partial match", "partialMatchingImages"), ("visually similar", "visuallySimilarImages"))
        for i in web.get(key) or [] if i.get("url")
    ]
    return _finish("cloud_vision", best, pages, exact, visual)


def _strip_tags(text: str) -> str:
    # Vision wraps matched words in <b>...</b> in page titles.
    return text.replace("<b>", "").replace("</b>", "")


# --- backends ----------------------------------------------------------------------------------

class Backend(Protocol):
    name: str

    async def search(self, jpeg: bytes) -> ReverseSearchResult: ...


def fit_jpeg(jpeg: bytes, max_bytes: int) -> bytes:
    """Re-encodes a JPEG to fit max_bytes: lower quality first, then smaller size."""
    if len(jpeg) <= max_bytes:
        return jpeg
    with Image.open(io.BytesIO(jpeg)) as src:
        img = src.convert("RGB")
    while True:
        for quality in (85, 75, 65):
            buf = io.BytesIO()
            img.save(buf, "JPEG", quality=quality)
            if buf.tell() <= max_bytes:
                return buf.getvalue()
        w, h = img.size
        if max(w, h) < 64:
            raise BackendError("Could not compress the crop under the upload size limit")
        img = img.resize((max(1, int(w * 0.8)), max(1, int(h * 0.8))), Image.Resampling.LANCZOS)


def _error_text(response: httpx.Response) -> str:
    try:
        body = response.json()
    except ValueError:
        return response.text[:200]
    err = body.get("error") if isinstance(body, dict) else None
    if isinstance(err, dict):
        return str(err.get("message") or err)
    return str(err or body)[:300]


class SerpApiBackend:
    name = "serpapi"

    def __init__(self, api_key: str, transport: httpx.AsyncBaseTransport | None = None):
        self._key = api_key
        self._transport = transport

    async def search(self, jpeg: bytes) -> ReverseSearchResult:
        upload = await asyncio.to_thread(fit_jpeg, jpeg, SERPAPI_MAX_UPLOAD_BYTES)
        async with httpx.AsyncClient(transport=self._transport) as client:
            image_id = await self._upload(client, upload)
            data = await self._lens(client, image_id)
        return normalize_serpapi(data)

    async def _upload(self, client: httpx.AsyncClient, jpeg: bytes) -> str:
        try:
            r = await client.post(
                SERPAPI_UPLOAD_URL,
                data={"api_key": self._key},
                files={"image": ("crop.jpg", jpeg, "image/jpeg")},
                timeout=UPLOAD_TIMEOUT_S,
            )
        except httpx.TimeoutException as exc:
            raise BackendError(f"SerpApi image upload timed out after {UPLOAD_TIMEOUT_S} s; try again later.") from exc
        except httpx.HTTPError as exc:
            raise BackendError(f"Could not reach SerpApi ({type(exc).__name__}); check the network and try again.") from exc
        self._raise_for_status(r)
        body = r.json()
        if body.get("error") or not body.get("image_id"):
            raise BackendError(f"SerpApi rejected the image upload: {body.get('error') or body}")
        return str(body["image_id"])

    async def _lens(self, client: httpx.AsyncClient, image_id: str) -> dict:
        params = {"engine": "google_lens", "image_id": image_id, "type": "all", "api_key": self._key}
        try:
            r = await client.get(SERPAPI_SEARCH_URL, params=params, timeout=SEARCH_TIMEOUT_S)
        except httpx.TimeoutException as exc:
            raise BackendError(f"SerpApi Google Lens search timed out after {SEARCH_TIMEOUT_S} s; try again later.") from exc
        except httpx.HTTPError as exc:
            raise BackendError(f"Could not reach SerpApi ({type(exc).__name__}); check the network and try again.") from exc
        self._raise_for_status(r)
        data = r.json()
        error = data.get("error")
        status = (data.get("search_metadata") or {}).get("status")
        if error and not _is_empty_result(error):
            raise BackendError(f"SerpApi search failed: {error}")
        if status == "Error" and not error:
            raise BackendError("SerpApi search failed with status Error; try again later.")
        return data

    @staticmethod
    def _raise_for_status(r: httpx.Response) -> None:
        if r.is_success:
            return
        detail = _error_text(r)
        if r.status_code == 401:
            raise BackendError(f"SerpApi rejected the API key ({detail}). Check {SERPAPI_ENV}.")
        if r.status_code == 403:
            raise BackendError(f"SerpApi refused the request ({detail}). Check the SerpApi account.")
        if r.status_code == 429:
            if "run out of searches" in detail.lower():
                raise BackendError(
                    f"SerpApi account has run out of searches ({detail}). "
                    f"Upgrade the plan, wait for the monthly reset, or use backend='cloud_vision' with {VISION_ENV}."
                )
            raise BackendError(f"SerpApi hourly rate limit reached ({detail}). Wait and try again later.")
        if r.status_code >= 500:
            raise BackendError(f"SerpApi server error {r.status_code} ({detail}); try again later.")
        raise BackendError(f"SerpApi request failed with HTTP {r.status_code}: {detail}")


def _is_empty_result(error: str) -> bool:
    # SerpApi reports "no results" through the error key with status Success.
    return "hasn't returned any results" in error or "no results" in error.lower()


class CloudVisionBackend:
    name = "cloud_vision"

    def __init__(self, api_key: str, transport: httpx.AsyncBaseTransport | None = None):
        self._key = api_key
        self._transport = transport

    async def search(self, jpeg: bytes) -> ReverseSearchResult:
        body = {"requests": [{
            "image": {"content": base64.b64encode(jpeg).decode()},
            "features": [{"type": "WEB_DETECTION", "maxResults": VISION_MAX_RESULTS}],
        }]}
        try:
            async with httpx.AsyncClient(transport=self._transport) as client:
                r = await client.post(VISION_URL, params={"key": self._key}, json=body, timeout=SEARCH_TIMEOUT_S)
        except httpx.TimeoutException as exc:
            raise BackendError(f"Cloud Vision timed out after {SEARCH_TIMEOUT_S} s; try again later.") from exc
        except httpx.HTTPError as exc:
            raise BackendError(f"Could not reach Cloud Vision ({type(exc).__name__}); check the network.") from exc
        if not r.is_success:
            detail = _error_text(r)
            if r.status_code == 400 and "api key" in detail.lower():
                raise BackendError(f"Cloud Vision rejected the API key ({detail}). Check {VISION_ENV}.")
            if r.status_code == 403:
                raise BackendError(
                    f"Cloud Vision denied the request ({detail}). Check that the Vision API is enabled, "
                    "billing is active and the key is allowed to call it."
                )
            if r.status_code == 429:
                raise BackendError(f"Cloud Vision quota exhausted ({detail}). Wait or raise the quota.")
            raise BackendError(f"Cloud Vision request failed with HTTP {r.status_code}: {detail}")
        data = r.json()
        err = (data.get("responses") or [{}])[0].get("error")
        if err:
            raise BackendError(f"Cloud Vision could not process the image: {err.get('message', err)}")
        return normalize_vision(data)


# --- backend selection ---------------------------------------------------------------------------

def select_backend(
    requested: BackendName,
    env: Mapping[str, str],
    transport: httpx.AsyncBaseTransport | None = None,
) -> tuple[Backend, str | None]:
    """Returns (backend, note). The note explains a fallback, if one happened."""
    serp_key = env.get(SERPAPI_ENV, "").strip()
    vision_key = env.get(VISION_ENV, "").strip()
    if requested == "serpapi":
        if serp_key:
            return SerpApiBackend(serp_key, transport), None
        if vision_key:
            return CloudVisionBackend(vision_key, transport), (
                f"SerpApi is not configured ({SERPAPI_ENV} is not set), "
                "so Google Cloud Vision Web Detection was used instead."
            )
    elif requested == "cloud_vision":
        if vision_key:
            return CloudVisionBackend(vision_key, transport), None
        if serp_key:
            raise NotConfiguredError(f"{VISION_ENV} is not set. SerpApi is configured: use backend='serpapi'.")
    else:
        raise NotConfiguredError(f"Unknown backend {requested!r}; use 'serpapi' or 'cloud_vision'.")
    raise NotConfiguredError(f"No reverse image search backend is configured. {CONFIG_HELP}")


# --- cache -------------------------------------------------------------------------------------

class ResultCache:
    """In-memory LRU keyed by (backend, SHA-256 of the crop bytes), so a crop is never searched twice."""

    def __init__(self, max_entries: int = 256):
        self._data: OrderedDict[tuple[str, str], ReverseSearchResult] = OrderedDict()
        self._max = max_entries

    @staticmethod
    def key(backend: str, jpeg: bytes) -> tuple[str, str]:
        return backend, hashlib.sha256(jpeg).hexdigest()

    def get(self, key: tuple[str, str]) -> ReverseSearchResult | None:
        if key in self._data:
            self._data.move_to_end(key)
            return self._data[key]
        return None

    def put(self, key: tuple[str, str], value: ReverseSearchResult) -> None:
        self._data[key] = value
        self._data.move_to_end(key)
        while len(self._data) > self._max:
            self._data.popitem(last=False)


CACHE = ResultCache()


# --- orchestration -----------------------------------------------------------------------------

FaceFraction = Callable[[Image.Image], float]


def prepare_crop(data: bytes, region: tuple[float, float, float, float]) -> tuple[Image.Image, tuple[int, int, int, int]]:
    """Crops at original resolution and enforces the minimum crop size."""
    try:
        crop, box, _ = zoom.crop_pixels(data, region)
    except ValueError as exc:
        raise CropRejectedError(str(exc)) from exc
    w, h = crop.size
    if w < MIN_CROP_PX or h < MIN_CROP_PX:
        raise CropRejectedError(
            f"The region is {w}x{h} px in the original image; reverse search needs at least "
            f"{MIN_CROP_PX} px on each side. Choose a larger region that contains the whole object "
            "with some context around it."
        )
    return crop, box


def encode_crop(crop: Image.Image) -> bytes:
    """Canonical JPEG of the crop (longest side at most zoom.MAX_SIDE, never upscaled)."""
    img = crop.copy()
    img.thumbnail((zoom.MAX_SIDE, zoom.MAX_SIDE), Image.Resampling.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=90)
    return buf.getvalue()


async def search_region(
    data: bytes,
    region: tuple[float, float, float, float],
    backend: Backend,
    face_fraction: FaceFraction,
    *,
    note: str | None = None,
    cache: ResultCache = CACHE,
) -> tuple[ReverseSearchResult, dict[str, Any]]:
    """Crops, applies the guardrails, then searches (or reuses a cached result)."""
    crop, box = await asyncio.to_thread(prepare_crop, data, region)
    fraction = await asyncio.to_thread(face_fraction, crop)
    if fraction > FACE_MAX_FRACTION:
        raise CropRejectedError(
            f"A face covers about {fraction:.0%} of this region. Reverse image search must not be used "
            "on people. Choose a region with the landmark, building, logo or sign only."
        )
    jpeg = await asyncio.to_thread(encode_crop, crop)
    key = cache.key(backend.name, jpeg)
    cached = cache.get(key)
    if cached is None:
        result = await backend.search(jpeg)
        cache.put(key, result)
    else:
        result = cached
    out = ReverseSearchResult(**{**asdict(result), "note": note})
    info = {"region": list(region), "source_pixels": list(box), "cached": cached is not None}
    return out, info
