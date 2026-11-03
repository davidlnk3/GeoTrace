from mcp.server.mcpserver import Context
from geotrace.server import mcp
from geotrace.core.ssis import ssis
from geotrace.core.images import load_image
from geotrace.core.calibration import calibrate
from geotrace.core.shadow_geometry import analyze_pair

import asyncio
from pydantic import Field
from typing import Annotated
from dataclasses import asdict

@mcp.tool(
    name="relative_sun_position_from_shadows",
    description="Returns the shadows information related to the objects from an image. "
    "It is precise, but only if confidence in the shadow is high. "
    "It provides confidence (pair_score), sun relative direction from the camera clockwise "
    "seen (0 = away from the camera, 90 = right), "
    "independent of the camera roll (sun_relative_azimuth_to_camera_deg), and sun elevation degrees "
    "(sun_altitude_deg). The camera is calibrated with GeoCalib and returned in 'camera', "
    "and the sun elevation is computed from the calibrated camera geometry."
)
async def relative_sun_position_from_shadows(
    image: Annotated[str, Field(description="URL or path of the image.")],
    ctx: Context
) -> dict:
    await ctx.info("Analizing the image with SSIS")
    result = await ssis.detect(image)
    pairs = [p for p in result["pairs"] if "object" in p and "shadow" in p]

    await ctx.info("Calibrating the camera with GeoCalib")
    data = await load_image(image)
    camera, K, down = await asyncio.to_thread(calibrate, data)

    geometries, skipped = [], []
    for p in pairs:
        try:
            geometries.append(analyze_pair(
                p["object"]["polygons"], p["shadow"]["polygons"],
                p["object"]["score"], p["shadow"]["score"],
                K=K, gravity=down,
            ))
        except ValueError as e:  # e.g. a base or tip above the horizon
            skipped.append({"pair_id": p.get("pair_id"), "reason": str(e)})
    geometries.sort(key=lambda g: g.pair_score, reverse=True)
    await ctx.info(f"{len(geometries)} object-shadow pairs detected, {len(skipped)} skipped")

    return {
        "num_pairs": len(geometries),
        "pairs": [asdict(g) for g in geometries],
        "skipped": skipped,
    }
