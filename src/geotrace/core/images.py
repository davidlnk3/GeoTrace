import asyncio
import io
import httpx
from PIL import Image, ExifTags, ImageFile
from PIL.TiffImagePlugin import IFDRational
from pathlib import Path
from dataclasses import dataclass

ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".heic", ".webp"}
MAX_BYTES = 20 * 1024 * 1024  # 20 MB

@dataclass
class ImageMetadata:
    format: str | None
    size: tuple[int, int]
    exif: dict
    exif_details: dict
    gps: dict

async def load_image(source: str) -> bytes:
    """Allows either an URL or a local path. Returns the image bytes."""
    if source.startswith(("http://", "https://")):
        return await _download_image(source)
    return await asyncio.to_thread(_read_local_file, source)

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

def read_metadata(data: bytes) -> dict:
    """Reads the EXIF metadata from bytes and returns it in JSON format"""
    with Image.open(io.BytesIO(data)) as image:
        exif = image.getexif()
        return ImageMetadata(
            format = image.format,
            size = image.size,
            exif = {ExifTags.TAGS.get(k, k): _clean(v)
                     for k, v in exif.items() if k not in (0x8769, 0x8825)},
            exif_details = {ExifTags.TAGS.get(k, k): _clean(v)
                             for k, v in exif.get_ifd(ExifTags.IFD.Exif).items()},
            gps = {ExifTags.GPSTAGS.get(k, k): _clean(v)
                    for k, v in exif.get_ifd(ExifTags.IFD.GPSInfo).items()}
        )

def _clean(v):
    """Transforms EXIF values to types that are convertible to JSON."""
    if isinstance(v, IFDRational):
        return float(v) if v.denominator else None
    if isinstance(v, bytes):
        return v.decode(errors="replace").strip("\x00")
    if isinstance(v, (tuple, list)):
        return [_clean(x) for x in v]
    return v