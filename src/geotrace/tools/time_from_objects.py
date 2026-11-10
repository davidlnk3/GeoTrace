from mcp.server.mcpserver import Context
from geotrace.server import mcp
from geotrace.core.images import load_image
from geotrace.core.photo_dating import estimate_capture_year
from geotrace.core.time_objects import time_from_objects as estimate_intervals

import asyncio
from pydantic import Field
from typing import Annotated


@mcp.tool(
    name="time_from_objects",
    description="Dates a photo in two independent ways. First, the interval in which each named "
    "object was built, from its likely build year until today, using EVA02-CLIP time probing "
    "(arXiv:2510.19559): the photo cannot be older than the most recent object in it. Second, the "
    "year the photo itself looks like it was taken, from photographic style, using a logistic year "
    "classifier over OpenCLIP embeddings (arXiv:2310.06633). "
    "objects is required: list the objects to date, for example after looking at the image. "
    "Returns current_year, image_taken_not_before_year (the latest built_from_year), for each "
    "object built_from_year (the estimated build year minus the paper's accuracy tolerance, "
    "5 to 20 years), built_until_year (today) and tolerance_years, and capture_year_estimate with "
    "estimated_year, expected_year, interval_80pct, confidence and model_year_range. "
    "The two signals are independent: the objects give a lower bound, the style estimate gives a "
    "likely era. Read capture_year_estimate with care, because that classifier can only answer "
    "within its trained year range and will do so even for a photo from outside it. "
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

    await ctx.info("Estimating when the photo was taken from its photographic style")
    try:
        dating = await asyncio.to_thread(estimate_capture_year, data)
    except Exception as exc:
        # The style classifier needs the 'geoclip' extra and its weights from the Hub. The object
        # intervals above do not, so a failure here is reported rather than failing the call.
        result["capture_year_error"] = f"{type(exc).__name__}: {exc}"
        await ctx.info(f"Style estimate could not run ({exc}); returning the object intervals only")
        return result

    result["capture_year_estimate"] = dating
    await ctx.info(
        f"The photo looks like it was taken in {dating['estimated_year']} "
        f"(80% interval {dating['interval_80pct'][0]}-{dating['interval_80pct'][1]})"
    )

    # The two signals can flatly disagree: if an object dates the photo after the classifier's last
    # year class, the style estimate cannot be right and the model should not average the two.
    if result["image_taken_not_before_year"] > dating["model_year_range"][1]:
        dating["disagrees_with_objects"] = (
            f"An object in this photo dates it to {result['image_taken_not_before_year']} or later, "
            f"which is outside this classifier's range, so ignore estimated_year here."
        )

    return result
