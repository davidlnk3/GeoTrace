"""Region selection, cropping and grid overlay for close inspection of an image.

All coordinates are fractions (0-1) of the image *after* EXIF orientation is applied,
so they match what view_image shows the model.
"""
import io
import re
from dataclasses import dataclass

from PIL import Image, ImageDraw, ImageFont

from geotrace.core import images

MAX_SIDE = 1568            # longest side of any image returned to the model
MAX_UPSCALE = 4.0          # never enlarge a crop more than this
MIN_REGION = 0.02          # minimum fraction of each side
GRID_COLS = 4              # one constant for the grid drawn and for cell_to_region,
GRID_ROWS = 4              # so the labels the model sees always map to the same region

_CELL_RE = re.compile(r"^\s*([A-Za-z])\s*([0-9]+)\s*$")


@dataclass(frozen=True)
class CropInfo:
    source_pixels: tuple[int, int, int, int]   # (left, top, right, bottom) in the original image
    source_size: tuple[int, int]               # (width, height) of the crop in the original image
    image_size: tuple[int, int]                # (width, height) of the original image
    scale: float
    upscaled: bool

    def as_dict(self) -> dict:
        return {
            "source_pixels": list(self.source_pixels),
            "source_size": list(self.source_size),
            "image_size": list(self.image_size),
            "scale": self.scale,
            "upscaled": self.upscaled,
        }


def _open_oriented(data: bytes) -> Image.Image:
    # EXIF-oriented RGB; raises ValueError if the bytes are not a decodable image.
    return images.open_rgb(data)


def _to_jpeg(img: Image.Image, quality: int = 90) -> bytes:
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=quality)
    return buf.getvalue()


def validate_region(x1: float, y1: float, x2: float, y2: float) -> None:
    if not (0 <= x1 < x2 <= 1 and 0 <= y1 < y2 <= 1):
        raise ValueError("Coordinates must satisfy 0 <= x1 < x2 <= 1 and 0 <= y1 < y2 <= 1")
    if x2 - x1 < MIN_REGION or y2 - y1 < MIN_REGION:
        raise ValueError(f"Region too small (min {MIN_REGION:.0%} of each side)")


def region_pixels(size: tuple[int, int], region: tuple[float, float, float, float]) -> tuple[int, int, int, int]:
    W, H = size
    x1, y1, x2, y2 = region
    return round(x1 * W), round(y1 * H), round(x2 * W), round(y2 * H)


def crop_pixels(data: bytes, region: tuple[float, float, float, float]) -> tuple[Image.Image, tuple[int, int, int, int], tuple[int, int]]:
    """Crops at original resolution. Returns (crop, pixel box, original size)."""
    validate_region(*region)
    img = _open_oriented(data)
    box = region_pixels(img.size, region)
    return img.crop(box), box, img.size


def crop_region(data: bytes, x1: float, y1: float, x2: float, y2: float) -> tuple[bytes, CropInfo]:
    """x1..y2 are fractions (0-1) of width and height. Returns (JPEG bytes, info).

    The crop is taken from the original image, then fitted to MAX_SIDE: shrunk if it is
    larger, enlarged (at most MAX_UPSCALE) if it is smaller.
    """
    crop, box, size = crop_pixels(data, (x1, y1, x2, y2))
    w, h = crop.size
    scale = min(MAX_SIDE / max(w, h), MAX_UPSCALE)
    if scale != 1:
        # Classic LANCZOS interpolation only. Never use AI super-resolution here: it invents
        # plausible detail that was never captured, which is unacceptable for evidence.
        crop = crop.resize((max(1, round(w * scale)), max(1, round(h * scale))), Image.Resampling.LANCZOS)
    return _to_jpeg(crop), CropInfo(
        source_pixels=box, source_size=(w, h), image_size=size,
        scale=round(scale, 2), upscaled=scale > 1,
    )


def cell_label(col: int, row: int) -> str:
    return f"{chr(ord('A') + col)}{row + 1}"


def cell_to_region(cell: str, cols: int = GRID_COLS, rows: int = GRID_ROWS) -> tuple[float, float, float, float]:
    """'C2' -> (0.5, 0.25, 0.75, 0.5). The letter is the column, the number the row."""
    m = _CELL_RE.match(cell)
    if not m:
        raise ValueError(f"Invalid cell {cell!r}: use a column letter and a row number, e.g. 'B3'")
    col = ord(m.group(1).upper()) - ord("A")
    row = int(m.group(2)) - 1
    if not (0 <= col < cols and 0 <= row < rows):
        last = cell_label(cols - 1, rows - 1)
        raise ValueError(f"Cell {cell!r} is outside the grid (A1..{last})")
    return col / cols, row / rows, (col + 1) / cols, (row + 1) / rows


def resolve_region(cell: str | None, region: list[float] | tuple[float, ...] | None) -> tuple[float, float, float, float]:
    """Exactly one of cell or region. Returns a validated (x1, y1, x2, y2)."""
    if (cell is None) == (region is None):
        raise ValueError(
            "Give exactly one of 'cell' (a grid label such as 'C2', from view_image(grid=True)) "
            "or 'region' ([x1, y1, x2, y2] as fractions 0-1 of width and height)"
        )
    if cell is not None:
        return cell_to_region(cell)
    assert region is not None
    if len(region) != 4:
        raise ValueError("region must have 4 numbers: [x1, y1, x2, y2] as fractions 0-1")
    x1, y1, x2, y2 = (float(v) for v in region)
    validate_region(x1, y1, x2, y2)
    return x1, y1, x2, y2


def downscale(data: bytes, quality: int = 85) -> bytes:
    """Oriented, RGB JPEG fitting within MAX_SIDE (no upscaling)."""
    img = _open_oriented(data)
    img.thumbnail((MAX_SIDE, MAX_SIDE))
    return _to_jpeg(img, quality)


def draw_grid(data: bytes, cols: int = GRID_COLS, rows: int = GRID_ROWS) -> bytes:
    """Overlays a labelled grid (A1..D4) on a downscaled copy of the image."""
    img = _open_oriented(data)
    img.thumbnail((MAX_SIDE, MAX_SIDE))
    W, H = img.size
    draw = ImageDraw.Draw(img)
    line_w = max(2, round(min(W, H) / 200))
    font_size = max(12, round(min(W / cols, H / rows) / 6))
    font = ImageFont.load_default(size=font_size)
    yellow, black = (255, 220, 0), (0, 0, 0)

    for c in range(1, cols):
        x = round(c * W / cols)
        draw.line([(x, 0), (x, H)], fill=black, width=line_w + 2)
        draw.line([(x, 0), (x, H)], fill=yellow, width=line_w)
    for r in range(1, rows):
        y = round(r * H / rows)
        draw.line([(0, y), (W, y)], fill=black, width=line_w + 2)
        draw.line([(0, y), (W, y)], fill=yellow, width=line_w)

    pad = max(3, font_size // 4)
    for r in range(rows):
        for c in range(cols):
            x0, y0 = round(c * W / cols) + pad, round(r * H / rows) + pad
            draw.text((x0, y0), cell_label(c, r), font=font, fill=yellow,
                      stroke_width=max(1, font_size // 10), stroke_fill=black)
    return _to_jpeg(img, 85)
