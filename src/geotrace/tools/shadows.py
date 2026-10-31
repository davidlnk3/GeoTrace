from mcp.server.mcpserver import Context
from geotrace.server import mcp
from geotrace.core.ssis import ssis
from geotrace.core.shadow_geometry import analyze_pair

from gradio_client import handle_file
from pydantic import Field
from typing import Annotated
from dataclasses import asdict

@mcp.tool(
    name="detect_shadows",
    description="Returns the shadows information related to the objects from an image. "
    "It provides confidence (pair_score), "
    "shadow direction relative to the image being clockwise from 'up' (0 = up, 90 = right) which may be used later for azimuth calculation (shadow_direction_deg), "
    "and sun elevation degrees (estimated_sun_elevation_deg)."
)
async def detect_shadows(
    image: Annotated[str, Field(description="URL or path of the image.")],
    camera_tilt_deg: Annotated[float, Field(
        description="Angle of the camera's optical axis below the horizon, in degrees (0 = horizontal, 90 = straight down). "
        "Required. Used to correct the perspective foreshortening of the sun elevation.",
        gt=0, lt=90,
    )],
    ctx: Context,
) -> dict:
    await ctx.info("Analizing the image with SSIS")
    result = await ssis.detect(image)

    pairs = [
        analyze_pair(
            p["object"]["polygons"], p["shadow"]["polygons"],
            p["object"]["score"], p["shadow"]["score"],
            camera_tilt_deg=camera_tilt_deg,
        )
        for p in result["pairs"]
        if "object" in p and "shadow" in p
    ]
    pairs.sort(key=lambda g: g.pair_score, reverse=True)
    await ctx.info(f"{len(pairs)} object-shadow pairs detected")

    return {
        "num_pairs": len(pairs),
        "pairs": [asdict(p) for p in pairs],
    }