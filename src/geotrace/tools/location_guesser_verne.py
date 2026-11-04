from mcp.server.mcpserver import Context
from geotrace.server import mcp
from geotrace.core.images import load_image
from geotrace.core.location_regions import describe_location
from geotrace.core.plonk_location import DEFAULT_SAMPLES

import asyncio
from pydantic import Field
from typing import Annotated


@mcp.tool(
    name="location_guesser_verne",
    description="Estimates where an image was probably taken. It draws many candidate locations and " \
    "returns them as regions. Returns: regions, sorted by probability; each with name (nearest place, admin region), "
    "probability, center (latitude, longitude), radius_km and country_code. regions[0] is the most "
    "probable area, and its center is the most probable point. unassigned_probability: the share of "
    "samples outside the returned regions. by_country: each country's share of all samples. "
    "area_90_km2: area of the fewest 1-degree cells holding 90% of the samples. localizability: "
    "high, medium or low, from area_90_km2. "
    "This is an estimate from visual content, not a verified location. The first call downloads the "
    "model and is slow; on a CPU each sample costs about a second.",
)
async def location_guesser_verne(
    image: Annotated[str, Field(description="URL or path of the image.")],
    ctx: Context,
    num_samples: Annotated[
        int,
        Field(ge=20, le=500, description="How many candidate locations to draw. More gives steadier "
              "regions but costs more time. Default 100."),
    ] = DEFAULT_SAMPLES,
) -> dict:
    await ctx.info("Estimating where the image was taken with PLONK")
    data = await load_image(image)
    result = await asyncio.to_thread(describe_location, data, num_samples)
    if result["regions"]:
        top = result["regions"][0]
        await ctx.info(f"Most probable area: {top['name']} ({top['probability']})")
    return result
