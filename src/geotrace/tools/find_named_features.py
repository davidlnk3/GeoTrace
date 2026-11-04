from typing import Annotated

from mcp.server.mcpserver import Context
from mcp.server.mcpserver.exceptions import ToolError
from pydantic import BaseModel, Field

from geotrace.core import named_features as nf
from geotrace.core import overpass
from geotrace.server import mcp


class BoundingBox(BaseModel):
    south: float = Field(ge=-90, le=90, description="Southern latitude, decimal degrees.")
    west: float = Field(ge=-180, le=180, description="Western longitude, decimal degrees.")
    north: float = Field(ge=-90, le=90, description="Northern latitude, decimal degrees.")
    east: float = Field(ge=-180, le=180, description="Eastern longitude, decimal degrees.")


@mcp.tool(
    name="find_named_features",
    description="Searches OpenStreetMap for features (nodes, ways, relations) whose name contains the "
    "given text, inside a bounding box. Use it with place clues read by extract_text: shop, street, "
    "bus stop, station or place names. Checks name, name:* (e.g. name:en), alt_name, old_name and "
    "official_name. Matching is case-insensitive, ignores accents on common Latin letters and "
    "collapses whitespace, and matches part of a name (\"Farmacia Roma\" finds \"Antica Farmacia "
    "Roma\"). The text is matched literally, not as a regex. Fix obvious OCR misreadings before "
    "searching; a shorter, distinctive part of a name finds more. "
    f"The bbox may be at most {nf.MAX_BBOX_KM2:g} km² and must not cross the antimeridian. "
    "tags optionally filters by exact tag values, e.g. {\"amenity\": \"pharmacy\"}; use \"*\" to "
    "only require the key. Returns up to 25 results nearest to the bbox centre, each with name, "
    "matched_field, matched_value, osm_type, osm_id, osm_url, lat, lon (centre for ways and "
    "relations), distance_to_bbox_center_m and main tags; plus total_found and truncated. "
    "An empty result includes a note with suggestions. Data © OpenStreetMap contributors.",
)
async def find_named_features(
    name: Annotated[
        str,
        Field(description="Text to search for in feature names, e.g. a shop name read on a sign."),
    ],
    bbox: Annotated[BoundingBox, Field(description="Search area in decimal degrees.")],
    ctx: Context,
    tags: Annotated[
        dict[str, str] | None,
        Field(description="Optional exact tag filters, e.g. {\"shop\": \"bakery\"}. "
              "Use \"*\" as the value to require only the key."),
    ] = None,
) -> dict:
    box = nf.BBox(bbox.south, bbox.west, bbox.north, bbox.east)
    await ctx.info(f"Searching OpenStreetMap for names matching {name!r}")
    try:
        result = await nf.find_named_features(name, box, tags)
    except (nf.NamedFeatureError, overpass.OverpassError) as exc:
        raise ToolError(str(exc)) from exc
    await ctx.info(f"Found {result['total_found']} matching features")
    return result
