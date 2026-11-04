from mcp.server.mcpserver import Context
from geotrace.server import mcp
from geotrace.core.images import load_image
from geotrace.core.location_geoclip import location_guesser_geoclip as guess_location

import asyncio
from pydantic import Field
from typing import Annotated


@mcp.tool(
    name="location_guesser_geoclip",
    description="Returns the most probable location of the image, using the GeoCLIP model "
    "(arXiv:2309.16020). It scores the image against a gallery of 100K GPS coordinates and "
    "returns the top_k candidates, best first. Returns most_probable (the best candidate) and "
    "candidates, each with rank, latitude, longitude and probability. The probabilities rank "
    "the candidates; they are not calibrated confidences. The model guesses even when the image "
    "has no location clues, so check the scene (signs, architecture, vegetation, terrain) before "
    "trusting the guess.",
)
async def location_guesser_geoclip(
    image: Annotated[str, Field(description="URL or path of the image.")],
    ctx: Context,
    top_k: Annotated[
        int,
        Field(ge=1, le=50, description="Number of candidate locations to return. Defaults to 5."),
    ] = 5,
) -> dict:
    await ctx.info("Guessing the location of the image with GeoCLIP")
    data = await load_image(image)
    result = await asyncio.to_thread(guess_location, data, top_k)
    best = result["most_probable"]
    await ctx.info(f"Most probable: {best['latitude']}, {best['longitude']}")
    return result
