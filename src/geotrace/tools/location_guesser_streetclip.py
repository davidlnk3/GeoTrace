from mcp.server.mcpserver import Context
from geotrace.server import mcp
from geotrace.core.images import load_image
from geotrace.core.streetclip import locate

import asyncio
from pydantic import Field
from typing import Annotated


@mcp.tool(
    name="location_guesser_streetclip",
    description="Returns the most probable location of the image using StreetCLIP (geolocal/StreetCLIP), "
    "a CLIP model trained on street-view photos. Zero-shot: the image is scored against one prompt per "
    "candidate place. Without candidates, it chooses among about 200 countries and returns a "
    "representative latitude and longitude for each. With candidates, such as "
    "['Lisbon, Portugal', 'Porto, Portugal'], it chooses among those names, which have no coordinates. "
    "Returns most_probable and top (the top_k candidates), each with location, latitude, longitude and "
    "probability. Probabilities are relative to the candidate set. "
    "Use it as a first guess: it is a neural estimate and can be wrong, so check it against visual clues.",
)
async def location_guesser_streetclip(
    image: Annotated[str, Field(description="URL or path of the image.")],
    ctx: Context,
    candidates: Annotated[
        list[str] | None,
        Field(default=None, description="Optional places to choose from, for example "
              "['Lisbon, Portugal', 'Porto, Portugal']. Defaults to the built-in list of countries."),
    ] = None,
    top_k: Annotated[
        int,
        Field(default=5, ge=1, le=50, description="How many of the most probable candidates to return."),
    ] = 5,
) -> dict:
    await ctx.info("Scoring the image against candidate locations with StreetCLIP")
    data = await load_image(image)
    result = await asyncio.to_thread(locate, data, candidates, top_k)
    best = result["most_probable"]
    await ctx.info(f"Most probable location: {best['location']} ({best['probability']:.2%})")
    return result
