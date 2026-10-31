from mcp.server.mcpserver import Context
from geotrace.server import mcp
from geotrace.core import images

from pydantic import Field
from typing import Annotated

@mcp.tool(
    name="extract_metadata",
    description="Extracts EXIF metadata, camera data and GPS from an image given by a direct URL to it or a path from the user's computer. Ask for the file path if none is given.",
)
async def extract_metadata(
    image: Annotated[str, Field(description="URL or path of the image.")],
    ctx: Context
) -> images.ImageMetadata:
    return images.read_metadata(await images.load_image(image))