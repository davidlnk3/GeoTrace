import asyncio
from typing import Annotated

import httpx
from mcp.server.mcpserver import Context, Image
from mcp.server.mcpserver.exceptions import ToolError
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
    """Shared by zoom_image and reverse_search_region: exactly one of cell or region."""
    try:
        return zoom.resolve_region(cell, region)
    except ValueError as exc:
        raise ToolError(str(exc)) from exc


@mcp.tool(
    name="view_image",
    description="Returns the image so the model can describe its visual clues. "
    f"With grid=True it overlays a labelled {zoom.GRID_COLS}x{zoom.GRID_ROWS} grid "
    f"(columns A-{chr(ord('A') + zoom.GRID_COLS - 1)} left to right, rows 1-{zoom.GRID_ROWS} top to bottom) "
    "so you can name a cell for zoom_image or reverse_search_region. The grid is drawn on a copy; "
    "it is not part of the evidence.",
)
async def view_image(
    image: Annotated[str, Field(description="URL or path of the image.")],
    ctx: Context,
    grid: Annotated[bool, Field(description="Overlay a labelled grid (A1..D4) to pick regions.")] = False,
) -> Image:
    data = await load_image_or_error(image)
    try:
        if grid:
            jpeg = await asyncio.to_thread(zoom.draw_grid, data)
        else:
            jpeg = await asyncio.to_thread(zoom.downscale, data)
    except ValueError as exc:
        raise ToolError(str(exc)) from exc
    return Image(data=jpeg, format="jpeg")
