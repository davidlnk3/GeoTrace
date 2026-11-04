from mcp.server.mcpserver import Context
from geotrace.server import mcp
from geotrace.core.images import load_image
from geotrace.core.plonk_location import DEFAULT_CANDIDATES, guess_location

import asyncio
from pydantic import Field
from typing import Annotated


@mcp.tool(
    name="location_guesser_verne",
    description="Returns the most probable coordinates of an image, using PLONK (arXiv:2412.06781, "
    "model nicolas-dufour/PLONK_YFCC). The model draws several candidate locations from the image, "
    "and the one with the highest likelihood is returned. "
    "Returns latitude and longitude in decimal degrees, log_likelihood (higher is more probable; "
    "only compare it within one image), num_candidates and model. "
    "The first call downloads the model weights and is slow; on a CPU each candidate takes a few "
    "seconds. This is a guess from visual content, not a verified location.",
)
async def location_guesser_verne(
    image: Annotated[str, Field(description="URL or path of the image.")],
    ctx: Context,
    num_candidates: Annotated[
        int,
        Field(ge=1, le=64, description="How many candidate locations to sample before choosing "
              "the most probable one. More candidates cost more time. Default 16."),
    ] = DEFAULT_CANDIDATES,
) -> dict:
    await ctx.info("Estimating the location with PLONK")
    data = await load_image(image)
    result = await asyncio.to_thread(guess_location, data, num_candidates)
    await ctx.info(f"Most probable location: {result['latitude']}, {result['longitude']}")
    return result
