from mcp.server.mcpserver import Context, Image
from geotrace.server import mcp
from geotrace.core import images

import io

from pydantic import Field
from typing import Annotated
from PIL import Image as PILImage
from PIL import ImageOps

MAX_SIDE = 1568

@mcp.tool(
    name="view_image",
    description="Returns the image so the model can describe its visual clues"
)
async def view_image(
    image: Annotated[str, Field(description="URL or path of the image.")],
    ctx: Context
) -> Image:
    data = await images.load_image(image)
    with PILImage.open(io.BytesIO(data)) as img:
        img = ImageOps.exif_transpose(img).convert("RGB")
        img.thumbnail((MAX_SIDE, MAX_SIDE))
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=85)
    return Image(data=buf.getvalue(), format="jpeg")