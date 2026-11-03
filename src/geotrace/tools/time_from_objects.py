from mcp.server.mcpserver import Context
from geotrace.server import mcp
from geotrace.core.images import load_image
from geotrace.core.time_objects import time_from_objects as estimate_intervals

import asyncio
from pydantic import Field
from typing import Annotated


@mcp.tool(
    name="time_from_objects",
    description="Lists the human-made objects in an image and the interval in which each was built, "
    "from its earliest likely build year until today. Use it to narrow when the photo can have been taken: the photo cannot be "
    "older than the most recent object in it. "
    "Returns current_year, image_taken_not_before_year (the latest built_from_year), and for each "
    "object: most_likely_build_year, built_from_year (that year minus the paper's accuracy tolerance, "
    "5 to 20 years), built_until_year (today), tolerance_years, and presence (zero-shot probability, "
    "only when objects are detected). "
    "Give objects to skip detection and check specific objects, for example after looking at the "
    "image. Without objects it checks the TIME10k classes: car, mobile phone, ship, musical "
    "instrument, aircraft, weapon or ammunition. The estimates are approximate and only apply to "
    "human-made objects.",
)
async def time_from_objects(
    image: Annotated[str, Field(description="URL or path of the image.")],
    ctx: Context,
    objects: Annotated[
        list[str] | None,
        Field(description="Optional object names to date, for example ['bicycle', 'street lamp']. "
              "If omitted, the TIME10k classes are checked for presence."),
    ] = None,
) -> dict:
    await ctx.info("Estimating build-time intervals for the objects in the image")
    data = await load_image(image)
    result = await asyncio.to_thread(estimate_intervals, data, objects)
    await ctx.info(f"Image taken not before {result['image_taken_not_before_year']}")
    return result
