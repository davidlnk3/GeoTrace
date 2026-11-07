from mcp.server.mcpserver import Context
from mcp.server.mcpserver.exceptions import ToolError
from geotrace.server import mcp
from geotrace.core import extras
from geotrace.core.images import load_image
from geotrace.core.sun_model import estimate

import asyncio
from pydantic import Field
from typing import Annotated


@mcp.tool(
    name="relative_sun_position_from_outdoor_lighting",
    description="Estimates the sun's altitude above the horizon and its azimuth relative to the "
    "camera's optical axis from a single outdoor image, using a DINOv2 model trained on sun-free "
    "crops (the sun is assumed to be outside the frame). sun_altitude_deg: degrees above the horizon, "
    "0 to 90. sun_relative_azimuth_to_camera_deg: clockwise from the camera's forward direction, 0 = straight "
    "ahead, 90 = right, 180 = behind, in [0, 360). Uses the centre square of the image. "
    "Expected errors on validation data: median about 1.6 deg "
    "for altitude and 5.6 deg for azimuth, with a long tail of large azimuth errors.",
)
async def relative_sun_position_from_outdoor_lighting(
    image: Annotated[str, Field(description="URL or path of the image.")],
    ctx: Context,
) -> dict:
    try:
        extras.require_sun_extra()
    except extras.MissingExtraError as exc:
        raise ToolError(str(exc)) from exc
    await ctx.info("Estimating the sun position with the DINOv2 sun model")
    data = await load_image(image)
    result = await asyncio.to_thread(estimate, data)
    await ctx.info(
        f"Sun altitude {result['sun_altitude_deg']}°, "
        f"relative azimuth {result['sun_relative_azimuth_to_camera_deg']}°"
    )
    return result
