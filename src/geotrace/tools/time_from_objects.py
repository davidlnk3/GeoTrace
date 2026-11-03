from mcp.server.mcpserver import Context
from geotrace.server import mcp
from geotrace.core.images import load_image
from geotrace.core.time_objects import time_from_objects as estimate_intervals

import asyncio
from pydantic import Field
from typing import Annotated


@mcp.tool(
    name="time_from_objects",
    description="Gives the interval in which each named object was built, from its likely build year "
    "until today, using EVA02-CLIP time probing (arXiv:2510.19559). Use it to narrow when the photo "
    "can have been taken: the photo cannot be older than the most recent object in it. "
    "objects is required: list the objects to date, for example after looking at the image. "
    "Returns current_year, image_taken_not_before_year (the latest built_from_year), and for each "
    "object: most_likely_build_year, built_from_year (that year minus the paper's accuracy tolerance, "
    "5 to 20 years), built_until_year (today), and tolerance_years. "
    "The estimates are approximate and apply to human-made objects.",
)
async def time_from_objects(
    image: Annotated[str, Field(description="URL or path of the image.")],
    objects: Annotated[
        list[str],
        Field(min_length=1, description="Names of the objects to date, for example "
              "['bicycle', 'street lamp']. Required."),
    ],
    ctx: Context,
) -> dict:
    await ctx.info("Estimating build-time intervals for the objects in the image")
    data = await load_image(image)
    result = await asyncio.to_thread(estimate_intervals, data, objects)
    await ctx.info(f"Image taken not before {result['image_taken_not_before_year']}")
    return result
