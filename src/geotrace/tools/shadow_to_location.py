from mcp.server.mcpserver import Context
from geotrace.server import mcp
from geotrace.core.solar import find_locations, sun_position

from datetime import datetime, timezone
from pydantic import Field
from typing import Annotated


@mcp.tool(
    name="shadow_to_location",
    description="Given the date and time (UTC) of a photo and the sun's position (elevation and azimuth, "
    "as obtained from shadows detection), returns the probable locations (latitude, longitude) where the sun "
    "was in that position. Usually two mirror-image solutions exist; both are returned. "
    "Azimuth is compass degrees clockwise from north; a shadow points opposite the sun, "
    "so sun azimuth = shadow direction + 180 (mod 360).",
)
async def shadow_to_location(
    date: Annotated[str, Field(description="Date of the photo in UTC, as YYYY-MM-DD.", pattern=r"^\d{4}-\d{2}-\d{2}$")],
    time: Annotated[str, Field(description="Time of the photo in UTC, as HH:MM or HH:MM:SS.", pattern=r"^\d{2}:\d{2}(:\d{2})?$")],
    sun_elevation: Annotated[float, Field(description="Sun elevation above the horizon in degrees.", gt=0, le=90)],
    sun_azimuth: Annotated[float, Field(description="Sun azimuth in degrees, clockwise from north.", ge=0, lt=360)],
    ctx: Context,
) -> dict:
    when = datetime.fromisoformat(f"{date}T{time}").replace(tzinfo=timezone.utc)
    await ctx.info(f"Solving location for {when.isoformat()} with sun at {sun_elevation}° elevation, {sun_azimuth}° azimuth")

    candidates = []
    for lat, lon in find_locations(when, sun_elevation, sun_azimuth):
        pos = sun_position(lat, lon, when)  # self-check: the position the candidate would produce
        candidates.append({
            "latitude": round(lat, 4),
            "longitude": round(lon, 4),
            "sun_elevation_deg": round(pos.elevation_deg, 2),
            "sun_azimuth_deg": round(pos.azimuth_deg, 2),
        })
    await ctx.info(f"{len(candidates)} candidate locations found")

    return {
        "datetime_utc": when.isoformat(),
        "num_candidates": len(candidates),
        "candidates": candidates,
    }
