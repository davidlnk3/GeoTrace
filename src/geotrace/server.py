from mcp.server.mcpserver import Context, MCPServer

from pydantic import Field
import asyncio
import io
import httpx
from PIL import Image, ExifTags, ImageFile
from PIL.TiffImagePlugin import IFDRational
from pathlib import Path
from typing import Annotated

mcp = MCPServer("GeoTrace", log_level="ERROR")

ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".heic", ".webp"}

MAX_BYTES = 20 * 1024 * 1024  # 20 MB

async def _download_image(url: str) -> bytes:
    """Downloads an image from an url and returns its bytes"""
    async with httpx.AsyncClient(timeout=15, follow_redirects=True) as client:
        response = await client.get(url)
        response.raise_for_status()
    if len(response.content) > MAX_BYTES:
        raise ValueError("Image is too large")
    return response.content

def _read_local_file(filepath: str) -> bytes:
    """Reads an image from local device and returns its bytes"""
    path = Path(filepath).expanduser().resolve()
    if path.suffix.lower() not in ALLOWED_EXTENSIONS:
        raise ValueError(f"Not an image file: {path.name}")
    if path.stat().st_size > MAX_BYTES:
        raise ValueError("Image is too large")
    return path.read_bytes()

def _read_metadata(data: bytes) -> dict:
    """Reads the EXIF metadata from bytes and returns it in JSON format"""
    with Image.open(io.BytesIO(data)) as image:
        exif = image.getexif()
        return {
            "format": image.format,
            "size": image.size,
            "exif": {ExifTags.TAGS.get(k, k): _clean(v)
                     for k, v in exif.items() if k not in (0x8769, 0x8825)},
            "exif_details": {ExifTags.TAGS.get(k, k): _clean(v)
                             for k, v in exif.get_ifd(ExifTags.IFD.Exif).items()},
            "gps": {ExifTags.GPSTAGS.get(k, k): _clean(v)
                    for k, v in exif.get_ifd(ExifTags.IFD.GPSInfo).items()},
        }

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
    description="Downloads an image from a URL and extracts its EXIF metadata, "
                "camera data and GPS. Ask for the URL if none is given.",
)
async def extract_metadata_from_url(
    image_url: Annotated[str, Field(description="Direct link to the image file.")],
    ctx: Context,
) -> dict:
    data = await _download_image(image_url)
    metadata = _read_metadata(data)
    await ctx.info(f"Image {metadata['format']} {metadata['size']}, "
                   f"{len(metadata['exif'])} EXIF labels")
    return metadata


@mcp.tool(
    name="extract_metadata_from_filepath",
    description="Extracts EXIF metadata, camera data and GPS from an image on the "
                "user's computer. Ask for the file path if none is given.",
)
async def extract_metadata_from_filepath(
    image_filepath: Annotated[str, Field(description="Absolute path to the image file.")],
    ctx: Context,
) -> dict:
    data = await asyncio.to_thread(_read_local_file, image_filepath)
    metadata = _read_metadata(data)
    await ctx.info(f"Image {metadata['format']} {metadata['size']}")
    return metadata

# TODO: sun_elevation_from_shadow()

def main():
    mcp.run()

if __name__ == "__main__":
    main()

    