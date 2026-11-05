"""Start a GeoTrace server command over stdio and check that it lists every tool.

Usage: python scripts/check_install.py <command> [args...]
Example: python scripts/check_install.py uvx --python 3.12 --from . geotrace
"""
import asyncio
import sys

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

EXPECTED_TOOLS = {
    "view_image",
    "zoom_image",
    "extract_metadata",
    "extract_text",
    "find_named_features",
    "overpass_query",
    "location_guesser_yfcc",
    "location_guesser_osv",
    "location_guesser_inat",
    "reverse_search_image",
    "verify_candidate_locations",
    "compare_images",
    "relative_sun_position_from_outdoor_lighting",
    "relative_sun_position_from_shadows",
    "location_from_sun_position",
    "time_from_sun_position",
    "time_from_objects",
    # inspect_calibration (tools/calibration.py) is not imported in tools/__init__.py, so it isn't served.
}


async def main(command: list[str]) -> int:
    params = StdioServerParameters(command=command[0], args=command[1:])
    async with stdio_client(params) as (read, write), ClientSession(read, write) as session:
        await session.initialize()
        names = {tool.name for tool in (await session.list_tools()).tools}
    missing, unexpected = EXPECTED_TOOLS - names, names - EXPECTED_TOOLS
    print(f"{len(names)} tools listed")
    if missing or unexpected:
        print(f"missing: {sorted(missing)}\nunexpected: {sorted(unexpected)}")
        return 1
    return 0


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    sys.exit(asyncio.run(main(sys.argv[1:])))
