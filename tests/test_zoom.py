import io

import pytest
from PIL import Image

from geotrace.core import zoom


def make_jpeg(w: int, h: int, exif_orientation: int | None = None) -> bytes:
    img = Image.new("RGB", (w, h), (0, 0, 0))
    # Mark the top-left quadrant white so orientation and coordinates can be checked.
    img.paste((255, 255, 255), (0, 0, w // 2, h // 2))
    buf = io.BytesIO()
    if exif_orientation is None:
        img.save(buf, "JPEG", quality=95)
    else:
        exif = Image.Exif()
        exif[0x0112] = exif_orientation
        img.save(buf, "JPEG", quality=95, exif=exif)
    return buf.getvalue()


def open_jpeg(data: bytes) -> Image.Image:
    return Image.open(io.BytesIO(data))


# --- crop_region ---------------------------------------------------------------------------

def test_crop_maps_fractions_to_original_pixels():
    data = make_jpeg(4000, 3000)
    jpeg, info = zoom.crop_region(data, 0.5, 0.25, 0.75, 0.5)
    assert info.source_pixels == (2000, 750, 3000, 1500)
    assert info.source_size == (1000, 750)
    assert info.image_size == (4000, 3000)
    assert open_jpeg(jpeg).format == "JPEG"


def test_small_crop_is_upscaled_to_max_side():
    data = make_jpeg(4000, 3000)
    jpeg, info = zoom.crop_region(data, 0.5, 0.25, 0.75, 0.5)   # 1000 x 750
    assert info.upscaled
    assert info.scale == pytest.approx(zoom.MAX_SIDE / 1000, abs=0.01)
    assert max(open_jpeg(jpeg).size) == zoom.MAX_SIDE


def test_large_crop_is_downscaled_to_max_side():
    data = make_jpeg(6000, 4000)
    jpeg, info = zoom.crop_region(data, 0, 0, 1, 1)
    assert not info.upscaled
    assert info.scale < 1
    assert open_jpeg(jpeg).size == (zoom.MAX_SIDE, round(4000 * zoom.MAX_SIDE / 6000))


def test_upscale_is_capped():
    data = make_jpeg(1000, 1000)
    jpeg, info = zoom.crop_region(data, 0, 0, 0.1, 0.1)          # 100 x 100
    assert info.scale == zoom.MAX_UPSCALE
    assert open_jpeg(jpeg).size == (400, 400)


def test_crop_exactly_max_side_is_not_resized():
    data = make_jpeg(zoom.MAX_SIDE * 2, zoom.MAX_SIDE)
    jpeg, info = zoom.crop_region(data, 0, 0, 0.5, 1)
    assert info.scale == 1 and not info.upscaled
    assert open_jpeg(jpeg).size == (zoom.MAX_SIDE, zoom.MAX_SIDE)


def test_crop_respects_exif_orientation():
    # Stored 400x200 with orientation 6 (rotate 90° CW to display) -> displayed 200x400.
    data = make_jpeg(400, 200, exif_orientation=6)
    _, info = zoom.crop_region(data, 0, 0, 1, 1)
    assert info.image_size == (200, 400)


@pytest.mark.parametrize("region", [(0.5, 0, 0.5, 1), (0, 0.6, 1, 0.4), (-0.1, 0, 1, 1), (0, 0, 1.2, 1)])
def test_invalid_coordinates(region):
    with pytest.raises(ValueError, match="Coordinates"):
        zoom.crop_region(make_jpeg(100, 100), *region)


def test_region_too_small():
    with pytest.raises(ValueError, match="too small"):
        zoom.crop_region(make_jpeg(1000, 1000), 0, 0, 0.01, 0.5)


# --- grid and cells ------------------------------------------------------------------------

@pytest.mark.parametrize("cell,expected", [
    ("A1", (0, 0, 0.25, 0.25)),
    ("C2", (0.5, 0.25, 0.75, 0.5)),
    ("d4", (0.75, 0.75, 1, 1)),
    (" b3 ", (0.25, 0.5, 0.5, 0.75)),
])
def test_cell_to_region(cell, expected):
    assert zoom.cell_to_region(cell) == pytest.approx(expected)


@pytest.mark.parametrize("cell", ["E1", "A5", "A0", "AA1", "1A", "", "C"])
def test_cell_to_region_rejects_bad_cells(cell):
    with pytest.raises(ValueError):
        zoom.cell_to_region(cell)


def test_grid_labels_match_cell_to_region():
    # Every label drawn on the grid must be accepted by cell_to_region, and the cells tile the image.
    labels = [zoom.cell_label(c, r) for r in range(zoom.GRID_ROWS) for c in range(zoom.GRID_COLS)]
    assert labels[0] == "A1" and labels[-1] == "D4"
    area = sum((x2 - x1) * (y2 - y1) for x1, y1, x2, y2 in map(zoom.cell_to_region, labels))
    assert area == pytest.approx(1)


def test_draw_grid_is_downscaled_jpeg():
    out = open_jpeg(zoom.draw_grid(make_jpeg(4000, 2000)))
    assert out.format == "JPEG"
    assert out.size == (zoom.MAX_SIDE, zoom.MAX_SIDE // 2)


# --- cell / region mutual exclusion -------------------------------------------------------

def test_resolve_region_with_cell():
    assert zoom.resolve_region("B2", None) == pytest.approx((0.25, 0.25, 0.5, 0.5))


def test_resolve_region_with_region():
    assert zoom.resolve_region(None, [0.1, 0.2, 0.3, 0.4]) == (0.1, 0.2, 0.3, 0.4)


@pytest.mark.parametrize("cell,region", [(None, None), ("A1", [0, 0, 0.5, 0.5])])
def test_resolve_region_requires_exactly_one(cell, region):
    with pytest.raises(ValueError, match="exactly one"):
        zoom.resolve_region(cell, region)


def test_resolve_region_validates_region():
    with pytest.raises(ValueError, match="Coordinates"):
        zoom.resolve_region(None, [0.5, 0.5, 0.4, 0.9])
    with pytest.raises(ValueError, match="4 numbers"):
        zoom.resolve_region(None, [0.1, 0.2, 0.3])
