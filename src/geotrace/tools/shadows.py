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
    description="Gets the shadows related to the objects from an image."
)
async def detect_shadows(
    image: Annotated[str, Field(description="URL or path of the image.")],
    ctx: Context
) -> dict:
    await ctx.info("Analizing the image with SSIS")
    result = await ssis.detect(image)

    pairs = [
        analyze_pair(
            p["object"]["polygons"], p["shadow"]["polygons"],
            p["object"]["score"], p["shadow"]["score"],
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