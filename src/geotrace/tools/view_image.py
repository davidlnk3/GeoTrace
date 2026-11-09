import asyncio
import json
from typing import Annotated

import httpx
from mcp.server.mcpserver import Context, Image
from mcp.server.mcpserver.exceptions import ToolError
from mcp.types import TextContent
from pydantic import Field

from geotrace.core import images, zoom
from geotrace.server import mcp

CellArg = Annotated[
    str | None,
    Field(description="Grid cell from view_image(grid=True), e.g. 'C2' (letter = column, number = row). "
          "Give either cell or region, not both."),
]
RegionArg = Annotated[
    list[float] | None,
    Field(min_length=4, max_length=4,
          description="[x1, y1, x2, y2] as fractions 0-1 of width and height, with (0, 0) the top-left. "
          "Give either cell or region, not both."),
]


async def load_image_or_error(image: str) -> bytes:
    try:
        return await images.load_image(image)
    except (ValueError, OSError, httpx.HTTPError) as exc:
        raise ToolError(f"Could not load the image: {exc}") from exc


def region_or_error(cell: str | None, region: list[float] | None) -> tuple[float, float, float, float]:
    """Shared by view_image and reverse_search_image: exactly one of cell or region."""
    try:
        return zoom.resolve_region(cell, region)
    except ValueError as exc:
        raise ToolError(str(exc)) from exc


@mcp.tool(
    name="view_image",
    description="Returns the image so the model can describe its visual clues, either whole or "
    "zoomed into one region. "
    "Omit cell and region to see the whole image, downscaled. "
    f"With grid=True it overlays a labelled {zoom.GRID_COLS}x{zoom.GRID_ROWS} grid "
    f"(columns A-{chr(ord('A') + zoom.GRID_COLS - 1)} left to right, rows 1-{zoom.GRID_ROWS} top to bottom) "
    "so you can name a cell for a later call, or for extract_text or reverse_search_image. The grid "
    "is drawn on a copy; it is not part of the evidence. "
    "Give cell (a label from view_image(grid=True), e.g. 'C2') or region ([x1, y1, x2, y2] fractions "
    "0-1), but not both, to get that region cropped from the ORIGINAL full-resolution file, to read "
    "signs, plates on buildings, shop names or other small details. "
    f"Small crops are enlarged up to {zoom.MAX_UPSCALE:g}x with classic interpolation: an upscaled crop "
    "contains NO new detail, only bigger pixels. If text is still unreadable, report it as unreadable; "
    "never guess letters or numbers. A cropped call also returns JSON info: source_pixels (box in the "
    "original image), source_size, image_size, scale and upscaled.",
)
async def view_image(
    image: Annotated[str, Field(description="URL or path of the image.")],
    ctx: Context,
    grid: Annotated[bool, Field(description="Overlay a labelled grid (A1..D4) to pick regions. "
                                            "Only for the whole image.")] = False,
    cell: Annotated[
        str | None,
        Field(description="Show only this grid cell from view_image(grid=True), e.g. 'C2', cropped from "
              "the original file. Omit cell and region to see the whole image."),
    ] = None,
    region: Annotated[
        list[float] | None,
        Field(min_length=4, max_length=4,
              description="Show only [x1, y1, x2, y2] as fractions 0-1 of width and height, with (0, 0) "
              "the top-left, cropped from the original file. Omit cell and region to see the whole image."),
    ] = None,
) -> list[Image | TextContent]:
    if grid and not (cell is None and region is None):
        raise ToolError(
            "grid labels the whole image, so it cannot be combined with cell or region. Call "
            "view_image(grid=True) to pick a cell, then view_image(cell=...) to look at it."
        )
    box = None if cell is None and region is None else region_or_error(cell, region)
    data = await load_image_or_error(image)

    if box is None:
        try:
            jpeg = await asyncio.to_thread(zoom.draw_grid if grid else zoom.downscale, data)
        except ValueError as exc:
            raise ToolError(str(exc)) from exc
        return [Image(data=jpeg, format="jpeg")]

    x1, y1, x2, y2 = box
    try:
        jpeg, info = await asyncio.to_thread(zoom.crop_region, data, x1, y1, x2, y2)
    except ValueError as exc:
        raise ToolError(str(exc)) from exc
    payload = {"region": [x1, y1, x2, y2], **({"cell": cell.strip().upper()} if cell else {}), **info.as_dict()}
    return [Image(data=jpeg, format="jpeg"), TextContent(type="text", text=json.dumps(payload))]
