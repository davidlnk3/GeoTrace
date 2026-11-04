from mcp.server.mcpserver import Context
from mcp.server.mcpserver.exceptions import ToolError
from geotrace.server import mcp
from geotrace.core import overpass

from pydantic import Field
from typing import Annotated, Literal


@mcp.tool(
    name="overpass_query",
    description="Runs an Overpass QL query on the OpenStreetMap Overpass API inside a bounding box. "
    "The box size limit depends on rarity: rare 2500 km², medium 400 km², common 100 km² (default), "
    "very_common 10 km². Choose the rarity of the searched feature, not of the photo. "
    "Write the query body only (no [out:json] header and no out statement); "
    "the tool adds them. Put the placeholder {{bbox}} in every filter, for example "
    "node[\"amenity\"=\"cafe\"]({{bbox}}); The tool rejects queries without it. "
    f"timeout is in seconds (default {overpass.DEFAULT_TIMEOUT_S}, max {overpass.MAX_TIMEOUT_S}). "
    f"max_results caps the returned elements (default {overpass.DEFAULT_MAX_RESULTS}, "
    f"max {overpass.MAX_RESULTS_LIMIT}); count_total gives the full number and truncated says whether any were cut. "
    "Returns the elements as OSM JSON (type, id, lat/lon or tags, etc.).",
)
async def overpass_query(
    query: Annotated[
        str,
        Field(description="Overpass QL statements using {{bbox}} in each filter. "
              "Example: node[\"shop\"=\"bakery\"]({{bbox}});"),
    ],
    south: Annotated[float, Field(description="Southern latitude of the box, decimal degrees.")],
    west: Annotated[float, Field(description="Western longitude of the box, decimal degrees.")],
    north: Annotated[float, Field(description="Northern latitude of the box, decimal degrees.")],
    east: Annotated[float, Field(description="Eastern longitude of the box, decimal degrees.")],
    ctx: Context,
    timeout: Annotated[
        int,
        Field(ge=1, le=overpass.MAX_TIMEOUT_S,
              description=f"Server-side timeout in seconds. Default {overpass.DEFAULT_TIMEOUT_S}."),
    ] = overpass.DEFAULT_TIMEOUT_S,
    max_results: Annotated[
        int,
        Field(ge=1, le=overpass.MAX_RESULTS_LIMIT,
              description=f"Maximum number of elements to return. Default {overpass.DEFAULT_MAX_RESULTS}."),
    ] = overpass.DEFAULT_MAX_RESULTS,
    rarity: Annotated[
        Literal["rare", "medium", "common", "very_common"],
        Field(description="How common the searched feature is. Sets the maximum box size: "
              "rare 2500 km², medium 400 km², common 100 km², very_common 10 km². Default common."),
    ] = overpass.DEFAULT_RARITY,
) -> dict:
    await ctx.info("Querying the Overpass API")
    try:
        result = await overpass.run_query(
            query, south, west, north, east,
            timeout=timeout, max_results=max_results, rarity=rarity,
        )
    except overpass.OverpassError as exc:
        raise ToolError(str(exc)) from exc
    await ctx.info(f"Overpass returned {result['count_total']} elements")
    return result
