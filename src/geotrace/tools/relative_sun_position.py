from mcp.server.mcpserver import Context
from geotrace.server import mcp
from geotrace.core.ssis import ssis
from geotrace.core.images import load_image
from geotrace.core.calibration import calibrate
from geotrace.core.shadow_geometry import analyze_pair
from geotrace.core.sun_model import estimate

import asyncio
from pydantic import Field
from typing import Annotated
from dataclasses import asdict

SHADOW_CONFIDENCE = 0.8  # a pair scoring above this is trusted over the lighting model

SHADOW_KEYS = ("num_pairs", "pairs", "skipped")


async def _from_shadows(image: str, ctx: Context) -> dict:
    """Object-shadow geometry: SSIS pairs measured against the GeoCalib camera.

    Returns the pairs sorted by confidence (pair_score), highest first, plus the pairs whose
    geometry could not be solved.
    """
    await ctx.info("Analizing the image with SSIS")
    result = await ssis.detect(image)
    pairs = [p for p in result["pairs"] if "object" in p and "shadow" in p]

    await ctx.info("Calibrating the camera with GeoCalib")
    data = await load_image(image)
    _camera, K, down = await asyncio.to_thread(calibrate, data)

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


@mcp.tool(
    name="relative_sun_position",
    description="Estimates the sun's altitude above the horizon and its azimuth relative to the "
    "camera from a single outdoor image. It tries object-shadow geometry first (SSIS pairs measured "
    f"against a GeoCalib camera) and uses the best pair when its confidence (pair_score) is above "
    f"{SHADOW_CONFIDENCE}, because that is the precise method. Otherwise it falls back to a DINOv2 "
    "model trained on sun-free crops (the sun is assumed to be outside the frame), which always "
    "returns an answer but a rougher one. "
    "Returns: source ('shadows' or 'outdoor_lighting'), which says which method produced the "
    "numbers and so how much to trust them. sun_altitude_deg: degrees above the horizon, 0 to 90. "
    "sun_relative_azimuth_to_camera_deg: clockwise from the camera's forward direction, 0 = straight "
    "ahead, 90 = right, 180 = behind, in [0, 360); independent of the camera roll. With source "
    "'shadows' it also returns pair_score for the pair used. Whenever the shadow stage ran it "
    "returns its full detail: num_pairs, pairs (all pairs, best confidence first) and skipped; a "
    "fallback with pairs present therefore means shadows were found but none was confident enough. "
    "shadow_error appears if the shadow stage could not run at all. "
    "Expected errors for 'outdoor_lighting' on validation data: median about 1.6 deg for altitude "
    "and 5.6 deg for azimuth, with a long tail of large azimuth errors.",
)
async def relative_sun_position(
    image: Annotated[str, Field(description="URL or path of the image.")],
    ctx: Context,
) -> dict:
    shadows, shadow_error = None, None
    try:
        shadows = await _from_shadows(image, ctx)
    except Exception as exc:
        # The shadow stage needs the 'calibration' extra and a reachable SSIS Space. When either is
        # missing there is no confident pair, which is exactly the case the lighting model covers.
        shadow_error = f"{type(exc).__name__}: {exc}"
        await ctx.info(f"Shadow analysis could not run ({exc}); using the lighting model instead")

    if shadows and shadows["pairs"] and shadows["pairs"][0]["pair_score"] > SHADOW_CONFIDENCE:
        best = shadows["pairs"][0]
        await ctx.info(f"Using the shadow pair scoring {best['pair_score']}")
        return {
            "source": "shadows",
            "sun_altitude_deg": best["sun_altitude_deg"],
            "sun_relative_azimuth_to_camera_deg": best["sun_relative_azimuth_to_camera_deg"],
            "pair_score": best["pair_score"],
            **shadows,
        }

    await ctx.info("Estimating the sun position with the DINOv2 sun model")
    data = await load_image(image)
    result = await asyncio.to_thread(estimate, data)
    await ctx.info(
        f"Sun altitude {result['sun_altitude_deg']}°, "
        f"relative azimuth {result['sun_azimuth_relative_camera_deg']}°"
    )

    out = {
        "source": "outdoor_lighting",
        "sun_altitude_deg": result["sun_altitude_deg"],
        "sun_relative_azimuth_to_camera_deg": result["sun_azimuth_relative_camera_deg"],
    }
    if shadows:
        out |= {k: shadows[k] for k in SHADOW_KEYS}
    if shadow_error:
        out["shadow_error"] = shadow_error
    return out
