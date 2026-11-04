"""The MCP tools with the OCR backend and the Overpass client mocked."""

import asyncio
import io

import pytest
from mcp.server.mcpserver.exceptions import ToolError
from PIL import Image

from geotrace.core import ocr, overpass
from geotrace.tools import extract_text as extract_text_module
from geotrace.tools.extract_text import extract_text
from geotrace.tools.find_named_features import BoundingBox, find_named_features


class FakeContext:
    def __init__(self):
        self.messages = []

    async def info(self, message):
        self.messages.append(message)


def png_bytes():
    buf = io.BytesIO()
    Image.new("RGB", (800, 600), "white").save(buf, format="PNG")
    return buf.getvalue()


class FakeBackend:
    def __init__(self, languages):
        self.languages = languages

    def read(self, image):
        return [
            ([[100, 100], [220, 100], [220, 140], [100, 140]], "VIA", 0.95),
            ([[240, 100], [400, 100], [400, 140], [240, 140]], "ROMA", 0.85),
            ([[10, 500], [60, 500], [60, 510], [10, 510]], "noise", 0.1),
        ]


@pytest.fixture
def ocr_mocked(monkeypatch):
    async def fake_load(source):
        return png_bytes()

    monkeypatch.setattr(ocr, "is_available", lambda: True)
    monkeypatch.setattr(ocr, "_backend_factory", FakeBackend)
    monkeypatch.setattr(ocr, "_readers", {})
    monkeypatch.setattr(extract_text_module, "load_image", fake_load)


def test_extract_text(ocr_mocked):
    ctx = FakeContext()
    result = asyncio.run(extract_text(image="photo.jpg", ctx=ctx))  # type: ignore[arg-type]
    assert result["image_size"] == (800, 600)
    assert result["languages"] == ocr.DEFAULT_LANGUAGES
    assert [b["text"] for b in result["blocks"]] == ["VIA ROMA"]
    assert result["total_blocks"] == 1 and result["truncated"] is False
    assert any("download" in m for m in ctx.messages)

    # Second call: models already loaded, no download message.
    ctx2 = FakeContext()
    asyncio.run(extract_text(image="photo.jpg", ctx=ctx2))  # type: ignore[arg-type]
    assert not any("download" in m for m in ctx2.messages)


def test_extract_text_empty_has_note(ocr_mocked):
    result = asyncio.run(extract_text(image="x.jpg", ctx=FakeContext(), min_confidence=0.99))  # type: ignore[arg-type]
    assert result["blocks"] == [] and "note" in result


def test_extract_text_invalid_languages(ocr_mocked):
    with pytest.raises(ToolError, match="separate calls"):
        asyncio.run(extract_text(image="x.jpg", ctx=FakeContext(), languages=["ja", "ru"]))  # type: ignore[arg-type]


def test_extract_text_without_extra(monkeypatch):
    monkeypatch.setattr(ocr, "is_available", lambda: False)
    with pytest.raises(ToolError, match="uv sync --extra ocr"):
        asyncio.run(extract_text(image="x.jpg", ctx=FakeContext()))  # type: ignore[arg-type]


def test_extract_text_bad_image(ocr_mocked, monkeypatch):
    async def bad_load(source):
        raise FileNotFoundError(source)

    monkeypatch.setattr(extract_text_module, "load_image", bad_load)
    with pytest.raises(ToolError, match="Could not load"):
        asyncio.run(extract_text(image="missing.jpg", ctx=FakeContext()))  # type: ignore[arg-type]


BOX = BoundingBox(south=41.89, west=12.47, north=41.91, east=12.49)


def test_find_named_features(monkeypatch):
    queries = []
    payloads = [
        {"elements": [{"type": "count", "tags": {"total": "1"}}]},
        {"elements": [{"type": "node", "id": 7, "lat": 41.9, "lon": 12.48,
                       "tags": {"name": "Farmacia Pasteur", "amenity": "pharmacy"}}]},
    ]

    async def fake_run_raw(query, timeout=25):
        queries.append(query)
        return payloads.pop(0)

    monkeypatch.setattr(overpass, "run_raw", fake_run_raw)
    result = asyncio.run(find_named_features(
        name="farmacia pasteur", bbox=BOX, ctx=FakeContext(), tags={"amenity": "pharmacy"},  # type: ignore[arg-type]
    ))
    assert result["total_found"] == 1
    assert result["results"][0]["osm_id"] == 7
    assert '["amenity"="pharmacy"]' in queries[0]
    assert "(41.89,12.47,41.91,12.49)" in queries[0]


def test_find_named_features_errors_become_tool_errors(monkeypatch):
    async def failing(query, timeout=25):
        raise overpass.OverpassError("all Overpass endpoints failed")

    monkeypatch.setattr(overpass, "run_raw", failing)
    with pytest.raises(ToolError, match="endpoints failed"):
        asyncio.run(find_named_features(name="Roma", bbox=BOX, ctx=FakeContext()))  # type: ignore[arg-type]
    with pytest.raises(ToolError, match="at least"):
        asyncio.run(find_named_features(name="R", bbox=BOX, ctx=FakeContext()))  # type: ignore[arg-type]

