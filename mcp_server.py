from mcp.server.mcpserver import Context, MCPServer
from pydantic import Field

import io
import httpx
from PIL import Image, ExifTags
from PIL.TiffImagePlugin import IFDRational

mcp = MCPServer("GeoTrace", log_level="ERROR")

MAX_BYTES = 20 * 1024 * 1024  # 20 MB

def _clean(v):
    """Transforms EXIF values to types that are convertible to JSON."""
    if isinstance(v, IFDRational):
        return float(v) if v.denominator else None
    if isinstance(v, bytes):
        return v.decode(errors="replace").strip("\x00")
    if isinstance(v, (tuple, list)):
        return [_clean(x) for x in v]
    return v

@mcp.tool(
    name="extract_metadata_from_url",
    description="Extracts metadata of an image from a given url."
)

async def extract_metadata_from_url(
    ctx: Context,
    image_url: str = Field(description="Link to the image whose metadata is going to be extracted.")
) -> dict :
    async with httpx.AsyncClient(timeout=15, follow_redirects=True) as client:
        response = await client.get(image_url)
        response.raise_for_status()
    if len(response.content) > MAX_BYTES:
        raise ValueError("Image is too large")

    with Image.open(io.BytesIO(response.content)) as img:
        exif = img.getexif()
        await ctx.info(f"Image {img.format} {img.size}, {len(exif)} EXIF labels")
        return {
            "format": img.format,
            "size": img.size,
            "exif": {ExifTags.TAGS.get(k, k): _clean(v)
                     for k, v in exif.items() if k not in (0x8769, 0x8825)},
            "exif_details": {ExifTags.TAGS.get(k, k): _clean(v)
                             for k, v in exif.get_ifd(ExifTags.IFD.Exif).items()},
            "gps": {ExifTags.GPSTAGS.get(k, k): _clean(v)
                    for k, v in exif.get_ifd(ExifTags.IFD.GPSInfo).items()}
        }


    