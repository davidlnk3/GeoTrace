"""Tool-layer tests. geotrace.tools imports every tool, and some need numpy (not a declared
dependency of the base install), so these are skipped when it is missing."""
import base64
import io
import json

import pytest
from PIL import Image

pytest.importorskip("numpy")

from mcp.server.mcpserver.exceptions import ToolError
from mcp.types import CallToolResult, ImageContent, TextContent

import geotrace.tools  # noqa: F401
from geotrace.server import mcp


async def call(name: str, args: dict) -> CallToolResult:
    result = await mcp.call_tool(name, args)
    assert isinstance(result, CallToolResult)
    return result


@pytest.fixture
def image_path(tmp_path):
    path = tmp_path / "photo.jpg"
    Image.new("RGB", (4000, 3000), (10, 20, 30)).save(path, "JPEG")
    return str(path)


async def test_zoom_image_returns_image_and_info(image_path):
    result = await call("zoom_image", {"image": image_path, "cell": "c2"})
    assert not result.is_error
    image, text = result.content
    assert isinstance(image, ImageContent) and image.mime_type == "image/jpeg"
    assert isinstance(text, TextContent)
    info = json.loads(text.text)
    assert info["cell"] == "C2"
    assert info["source_pixels"] == [2000, 750, 3000, 1500]
    assert info["upscaled"] is True


@pytest.mark.parametrize("args", [{}, {"cell": "A1", "region": [0, 0, 0.5, 0.5]}])
async def test_zoom_image_requires_cell_xor_region(image_path, args):
    with pytest.raises(ToolError, match="exactly one"):
        await mcp.call_tool("zoom_image", {"image": image_path, **args})


async def test_view_image_grid(image_path):
    result = await call("view_image", {"image": image_path, "grid": True})
    (image,) = result.content
    assert isinstance(image, ImageContent)
    assert Image.open(io.BytesIO(base64.b64decode(image.data))).size == (1568, 1176)
