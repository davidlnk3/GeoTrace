"""Tool-layer tests for reverse_search_image. Backends and the face detector are patched,
so no network request is made. Skipped when numpy is missing (geotrace.tools needs it)."""
import json

import pytest
from PIL import Image

pytest.importorskip("numpy")

from mcp.server.mcpserver.exceptions import ToolError
from mcp.types import CallToolResult, TextContent

import geotrace.tools  # noqa: F401
from geotrace.core import face_guard
from geotrace.core import reverse_search as rs
from geotrace.server import mcp


@pytest.fixture
def image_path(tmp_path):
    path = tmp_path / "photo.jpg"
    Image.new("RGB", (2000, 1500), (10, 20, 30)).save(path, "JPEG")
    return str(path)


@pytest.fixture(autouse=True)
def isolated(monkeypatch):
    monkeypatch.delenv(rs.SERPAPI_ENV, raising=False)
    monkeypatch.delenv(rs.VISION_ENV, raising=False)
    monkeypatch.setattr(rs, "CACHE", rs.ResultCache())
    monkeypatch.setattr(face_guard, "opencv_available", lambda: True)
    monkeypatch.setattr(face_guard, "largest_face_fraction", lambda _img: 0.0)

    async def fake_serp(self, jpeg):
        return rs.ReverseSearchResult(backend="serpapi", best_guess=["serp"])

    async def fake_vision(self, jpeg):
        return rs.ReverseSearchResult(backend="cloud_vision", best_guess=["vision"])

    monkeypatch.setattr(rs.SerpApiBackend, "search", fake_serp)
    monkeypatch.setattr(rs.CloudVisionBackend, "search", fake_vision)


async def call(args: dict) -> dict:
    result = await mcp.call_tool("reverse_search_image", args)
    assert isinstance(result, CallToolResult)
    text = result.content[0]
    assert isinstance(text, TextContent)
    return json.loads(text.text)


async def test_missing_extra_explains_install(image_path, monkeypatch):
    monkeypatch.setattr(face_guard, "opencv_available", lambda: False)
    monkeypatch.setenv(rs.SERPAPI_ENV, "k")
    with pytest.raises(ToolError, match="uv sync --extra reverse-search"):
        await call({"image": image_path, "cell": "B2"})


async def test_nothing_configured(image_path):
    with pytest.raises(ToolError, match=f"{rs.SERPAPI_ENV}.*{rs.VISION_ENV}"):
        await call({"image": image_path, "cell": "B2"})


async def test_serpapi_default(image_path, monkeypatch):
    monkeypatch.setenv(rs.SERPAPI_ENV, "k")
    monkeypatch.setenv(rs.VISION_ENV, "v")
    out = await call({"image": image_path, "cell": "b2"})
    assert out["backend"] == "serpapi" and out["note"] is None
    assert out["cell"] == "B2" and out["source_pixels"] == [500, 375, 1000, 750]


async def test_fallback_note_in_result(image_path, monkeypatch):
    monkeypatch.setenv(rs.VISION_ENV, "v")
    out = await call({"image": image_path, "region": [0, 0, 0.5, 0.5]})
    assert out["backend"] == "cloud_vision"
    assert rs.SERPAPI_ENV in out["note"]


async def test_cell_xor_region(image_path, monkeypatch):
    monkeypatch.setenv(rs.SERPAPI_ENV, "k")
    with pytest.raises(ToolError, match="exactly one"):
        await call({"image": image_path})


async def test_face_refusal_is_tool_error(image_path, monkeypatch):
    monkeypatch.setenv(rs.SERPAPI_ENV, "k")
    monkeypatch.setattr(face_guard, "largest_face_fraction", lambda _img: 0.3)
    with pytest.raises(ToolError, match="must not be used on people"):
        await call({"image": image_path, "cell": "A1"})


async def test_small_region_is_tool_error(image_path, monkeypatch):
    monkeypatch.setenv(rs.SERPAPI_ENV, "k")
    with pytest.raises(ToolError, match="larger region"):
        await call({"image": image_path, "region": [0, 0, 0.05, 0.5]})


async def test_full_image_region(image_path, monkeypatch):
    monkeypatch.setenv(rs.SERPAPI_ENV, "k")
    out = await call({"image": image_path, "region": [0, 0, 1, 1]})
    assert out["source_pixels"] == [0, 0, 2000, 1500]
