from mcp.server.mcpserver import Context
from geotrace.server import mcp
from geotrace.core.solar import find_matching_times

from datetime import datetime, timezone
from pydantic import Field
from typing import Annotated


@mcp.tool(
    name="time_from_sun_position",
    description="Given a location and the sun's position (elevation and azimuth), returns the "
    "probable dates and times (UTC) when the sun was in that position. "
    "Results are sorted best match first.",
)
async def time_from_sun_position(
    latitude: Annotated[float, Field(description="Latitude in decimal degrees, north positive.", ge=-90, le=90)],
    longitude: Annotated[float, Field(description="Longitude in decimal degrees, east positive.", ge=-180, le=180)],
    sun_elevation: Annotated[float, Field(description="Sun elevation above the horizon in degrees.", gt=0, le=90)],
    sun_azimuth: Annotated[float, Field(description="Sun azimuth in degrees, clockwise from north.", ge=0, lt=360)],
    ctx: Context,
    year: Annotated[int | None, Field(description="Year to search. Defaults to the current year.", ge=1900, le=2100)] = None,
    tolerance: Annotated[float, Field(description="Max angular error in degrees to accept a match.", gt=0, le=10)] = 1.0,
    max_results: Annotated[int, Field(description="Maximum number of candidates to return.", ge=1, le=100)] = 10,
) -> dict:
    year = year or datetime.now(timezone.utc).year
    await ctx.info(f"Searching {year} for times with sun at {sun_elevation}° elevation, {sun_azimuth}° azimuth")

    matches = find_matching_times(
        latitude, longitude, sun_elevation, sun_azimuth, year, tolerance,
    )[:max_results]

    candidates = [
        {
            "datetime_utc": when.isoformat(),
            "date": when.date().isoformat(),
            "time_utc": when.strftime("%H:%M"),
            "sun_elevation_deg": round(pos.elevation_deg, 2),
            "sun_azimuth_deg": round(pos.azimuth_deg, 2),
            "error_deg": round(err, 3),
        }
        for when, pos, err in matches
    ]
    await ctx.info(f"{len(candidates)} candidate times found")

    return {
        "year": year,
        "num_candidates": len(candidates),
        "candidates": candidates,
    }
