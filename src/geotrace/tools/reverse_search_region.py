import os
from typing import Annotated

from mcp.server.mcpserver import Context
from mcp.server.mcpserver.exceptions import ToolError
from pydantic import Field

from geotrace.core import face_guard, reverse_search
from geotrace.server import mcp
from geotrace.tools.view_image import CellArg, RegionArg, load_image_or_error, region_or_error


@mcp.tool(
    name="reverse_search_region",
    description="Runs a reverse image search on one region of the image to find where a landmark, building, "
    "logo, monument or distinctive sign appears online. Use it ONLY for those; never on people or faces "
    "(crops with a large face are refused). "
    "Pick the region with cell (from view_image(grid=True), e.g. 'C2') or region ([x1, y1, x2, y2] fractions 0-1); "
    "give exactly one. Choosing a good crop: include the WHOLE object with a little context around it, pick "
    "something distinctive (a named facade, a unique sign, a logo, not a generic lamp post, tree or road), and "
    f"make sure it has enough original resolution (at least {reverse_search.MIN_CROP_PX} px per side in the "
    "original image; check with zoom_image first). Each search can cost paid credits; an identical crop is "
    "answered from cache. backend: 'serpapi' (Google Lens via SerpApi, default) or 'cloud_vision' "
    "(Google Cloud Vision Web Detection). Returns best_guess, pages, exact_matches, visual_matches "
    f"(at most {reverse_search.MAX_ITEMS} each), total_results, truncated, the backend used and an optional note. "
    "Results are leads to verify, not proof of location.",
)
async def reverse_search_region(
    image: Annotated[str, Field(description="URL or path of the image.")],
    ctx: Context,
    cell: CellArg = None,
    region: RegionArg = None,
    backend: Annotated[
        reverse_search.BackendName,
        Field(description="'serpapi' (Google Lens, default) or 'cloud_vision' (Cloud Vision Web Detection)."),
    ] = "serpapi",
) -> dict:
    box = region_or_error(cell, region)
    if not face_guard.opencv_available():
        raise ToolError(
            "reverse_search_region needs OpenCV for its face safeguard. Install the optional extra with "
            f"'{face_guard.INSTALL_HINT}' (or pip install 'geotrace[reverse-search]') and restart the server."
        )
    try:
        engine, note = reverse_search.select_backend(backend, os.environ)
        data = await load_image_or_error(image)
        result, info = await reverse_search.search_region(
            data, box, engine, face_guard.largest_face_fraction, note=note,
        )
    except reverse_search.ReverseSearchError as exc:
        raise ToolError(str(exc)) from exc
    return {**result.as_dict(), **info, **({"cell": cell.strip().upper()} if cell else {})}
