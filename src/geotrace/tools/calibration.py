from mcp.server.mcpserver import Context
from geotrace.server import mcp
from geotrace.core.images import load_image
from geotrace.core.calibration import raw_output

from pydantic import Field
from typing import Annotated

import asyncio


@mcp.tool(
    name="inspect_calibration",
    description="Runs GeoCalib on an image and returns its raw outputs: image size, focal length, principal point, "
    "field of view, gravity vector, roll and pitch (degrees), their uncertainties, and the up-field confidence "
    "statistics. Use it to check the values the calibration gives for a photo.",
)
async def inspect_calibration(
    image: Annotated[str, Field(description="URL or path of the image.")],
    ctx: Context,
) -> dict:
    await ctx.info("Running GeoCalib")
    data = await load_image(image)
    return await asyncio.to_thread(raw_output, data)
