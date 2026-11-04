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

    def read(self, image, rotated=False):
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


def test_extract_text_full_image_has_no_region_fields(ocr_mocked):
    result = asyncio.run(extract_text(image="photo.jpg", ctx=FakeContext()))  # type: ignore[arg-type]
    assert "region" not in result and "source_pixels" not in result and "cell" not in result


def test_extract_text_cell(ocr_mocked):
    # A1 of the 4x4 grid on 800x600 is the top-left 200x150 box: VIA (100-220) is cut at 200.
    result = asyncio.run(extract_text(image="photo.jpg", ctx=FakeContext(), cell=" a1 "))  # type: ignore[arg-type]
    assert result["cell"] == "A1"
    assert result["region"] == (0.0, 0.0, 0.25, 0.25)
    assert result["source_pixels"] == (0, 0, 200, 150)
    assert result["image_size"] == (800, 600)
    assert all(0 <= b["bbox"][2] <= 200 and b["bbox"][3] <= 150 for b in result["blocks"])


def test_extract_text_region_offsets_bbox(ocr_mocked):
    result = asyncio.run(extract_text(image="photo.jpg", ctx=FakeContext(), region=[0.5, 0.5, 1, 1]))  # type: ignore[arg-type]
    assert result["source_pixels"] == (400, 300, 800, 600)
    # The fake backend returns the same boxes for any input, so they land offset by (400, 300).
    assert result["blocks"][0]["bbox"] == (500, 400, 800, 440)
    assert result["scale"] == 1.0 and result["retried"] is False and result["rotated_text"] is False


def test_extract_text_cell_and_region_rejected(ocr_mocked):
    with pytest.raises(ToolError, match="exactly one"):
        asyncio.run(extract_text(image="x.jpg", ctx=FakeContext(), cell="A1", region=[0, 0, 1, 1]))  # type: ignore[arg-type]


@pytest.mark.parametrize("kwargs", [{"cell": "Z9"}, {"region": [0.8, 0, 0.2, 1]}])
def test_extract_text_bad_cell_or_region(ocr_mocked, kwargs):
    with pytest.raises(ToolError):
        asyncio.run(extract_text(image="x.jpg", ctx=FakeContext(), **kwargs))  # type: ignore[arg-type]


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



def test_extract_text_retry_adds_warning_note(ocr_mocked, monkeypatch):
    calls = []

    class EmptyFirst(FakeBackend):
        def read(self, image, rotated=False):
            calls.append(image.size)
            return [] if len(calls) == 1 else super().read(image, rotated)

    monkeypatch.setattr(ocr, "_backend_factory", EmptyFirst)
    result = asyncio.run(extract_text(image="x.jpg", ctx=FakeContext(), cell="B2"))  # type: ignore[arg-type]
    assert calls == [(200, 150), (800, 600)]  # native cell, then x4
    assert result["retried"] is True and result["blocks"]
    assert "zoom_image" in result["note"]
