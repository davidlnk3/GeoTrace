from mcp.server.mcpserver import Context
from geotrace.server import mcp
from geotrace.core.images import load_image
from geotrace.core.location_regions import describe_location
from geotrace.core.plonk_location import DEFAULT_SAMPLES, INAT_MODEL, OSV_MODEL, YFCC_MODEL

import asyncio
from pydantic import Field
from typing import Annotated, Literal

# Each kind picks the PLONK model trained on that kind of imagery, plus the wording for the log line.
KINDS = {
    "street": (OSV_MODEL, "street-level photo", "the OSV-5M PLONK model"),
    "general": (YFCC_MODEL, "image", "PLONK"),
    "nature": (INAT_MODEL, "nature photo", "the iNaturalist PLONK model"),
}


@mcp.tool(
    name="location_guesser",
    description="Estimates where a photo was probably taken. It draws many candidate locations and "
    "returns them as regions. Choose kind to match the image: 'street' for roads, streets, buildings "
    "and signs (StreetCLIP encoder, downloaded on first use); 'nature' for plants, animals and "
    "landscapes; 'general' for anything else. Pick the ONE kind that fits the image; running a second "
    "kind is worthwhile only if the first gives low localizability. "
    "Returns: regions, sorted by probability; each with name (nearest place, admin region), "
    "probability, center (latitude, longitude), radius_km and country_code. regions[0] is the most "
    "probable area, and its center is the most probable point. unassigned_probability: the share of "
    "samples outside the returned regions. by_country: each country's share of all samples. "
    "area_90_km2: area of the fewest 1-degree cells holding 90% of the samples. localizability: "
    "high, medium or low, from area_90_km2. "
    "This is an estimate from visual content, not a verified location. Use it as one more clue.",
)
async def location_guesser(
    image: Annotated[str, Field(description="URL or path of the image.")],
    kind: Annotated[
        Literal["street", "general", "nature"],
        Field(description="Which kind of imagery this is: 'street' (roads, streets, buildings, signs), "
              "'nature' (plants, animals, landscapes) or 'general' (anything else)."),
    ],
    ctx: Context,
    num_samples: Annotated[
        int,
        Field(ge=20, le=500, description="How many candidate locations to draw. More gives steadier "
              "regions but costs more time. Default 100."),
    ] = DEFAULT_SAMPLES,
) -> dict:
    model_id, subject, model_name = KINDS[kind]
    await ctx.info(f"Estimating where the {subject} was taken with {model_name}")
    data = await load_image(image)
    result = await asyncio.to_thread(describe_location, data, num_samples, model_id)
    if result["regions"]:
        top = result["regions"][0]
        await ctx.info(f"Most probable area: {top['name']} ({top['probability']})")
    return result
