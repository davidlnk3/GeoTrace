import asyncio
import json
from typing import Annotated

from mcp.server.mcpserver import Context, Image
from mcp.server.mcpserver.exceptions import ToolError
from mcp.types import TextContent
from pydantic import Field

from geotrace.core import zoom
from geotrace.server import mcp
from geotrace.tools.view_image import CellArg, RegionArg, load_image_or_error, region_or_error


@mcp.tool(
    name="zoom_image",
    description="Returns one region of the image, cropped from the ORIGINAL full-resolution file "
    "(view_image shows a downscaled copy), to read signs, plates on buildings, shop names or other small details. "
    "Pick the region with cell (a label from view_image(grid=True), e.g. 'C2') or with region "
    "([x1, y1, x2, y2] fractions 0-1); give exactly one. "
    f"Small crops are enlarged up to {zoom.MAX_UPSCALE:g}x with classic interpolation: an upscaled crop "
    "contains NO new detail, only bigger pixels. If text is still unreadable, report it as unreadable; "
    "never guess letters or numbers. Also returns JSON info: source_pixels (box in the original image), "
    "source_size, image_size, scale and upscaled.",
)
async def zoom_image(
    image: Annotated[str, Field(description="URL or path of the image.")],
    ctx: Context,
    cell: CellArg = None,
    region: RegionArg = None,
) -> list[Image | TextContent]:
    x1, y1, x2, y2 = region_or_error(cell, region)
    data = await load_image_or_error(image)
    try:
        jpeg, info = await asyncio.to_thread(zoom.crop_region, data, x1, y1, x2, y2)
    except ValueError as exc:
        raise ToolError(str(exc)) from exc
    payload = {"region": [x1, y1, x2, y2], **({"cell": cell.strip().upper()} if cell else {}), **info.as_dict()}
    return [Image(data=jpeg, format="jpeg"), TextContent(type="text", text=json.dumps(payload))]
