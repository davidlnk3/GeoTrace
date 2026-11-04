import asyncio
from dataclasses import asdict
from typing import Annotated

from mcp.server.mcpserver import Context
from mcp.server.mcpserver.exceptions import ToolError
from pydantic import Field

from geotrace.core import ocr
from geotrace.core.images import load_image
from geotrace.server import mcp
from geotrace.tools.view_image import region_or_error


@mcp.tool(
    name="extract_text",
    description="Detects and reads text in an image with OCR (EasyOCR): signs, shop names, street "
    "names, bus stop names, posters, house numbers. Returns blocks sorted with the largest, most "
    "confident text first (at most 50). Each block has text, confidence (0-1), bbox (x1, y1, x2, y2 "
    "in pixels of the original image) and relative_size (share of the area read; big text is usually a sign). "
    "By default the whole image is read. To read one place only, pass cell (a label from "
    "view_image(grid=True), e.g. 'C2') or region ([x1, y1, x2, y2] fractions 0-1); give at most one. "
    "The region is cropped from the ORIGINAL full-resolution file, so small or distant text (a street "
    "sign, a house number, a shop front) is often read better than in a whole-image call. "
    "Nearby words on the same line are merged, e.g. 'VIA' + 'ROMA' -> 'VIA ROMA'. "
    "OCR can misread characters (0/O, 1/l/I, 5/S, rn/m, dropped or wrong accents), so cross-check "
    "each text against what you see in the image before using it. Names of shops, streets, stops "
    "and places are the most useful clues: search them with find_named_features. "
    "languages: EasyOCR codes. Default is Latin-script languages (en, es, fr, de, it, pt, nl). "
    "If you see another script, pass it, e.g. ['ru', 'en'], ['ar', 'en'], ['ja', 'en'], "
    "['ko', 'en'], ['ch_sim', 'en'], ['ch_tra', 'en'], ['hi', 'en'], ['th', 'en']. Non-Latin "
    "scripts can only be combined with English, so use one call per script. "
    "The first call for a language set downloads and loads the OCR models, which is slow.",
)
async def extract_text(
    image: Annotated[str, Field(description="URL or path of the image.")],
    ctx: Context,
    languages: Annotated[
        list[str] | None,
        Field(description="EasyOCR language codes, e.g. ['en', 'es'] or ['ru', 'en']. "
              "Default: common Latin-script languages."),
    ] = None,
    min_confidence: Annotated[
        float,
        Field(ge=0.0, le=1.0, description="Discard detections below this confidence. Default 0.3."),
    ] = ocr.DEFAULT_MIN_CONFIDENCE,
    cell: Annotated[
        str | None,
        Field(description="Read only this grid cell from view_image(grid=True), e.g. 'C2'. "
              "Omit cell and region to read the whole image."),
    ] = None,
    region: Annotated[
        list[float] | None,
        Field(min_length=4, max_length=4,
              description="Read only [x1, y1, x2, y2] as fractions 0-1 of width and height, with (0, 0) "
              "the top-left. Omit cell and region to read the whole image."),
    ] = None,
) -> dict:
    if not ocr.is_available():
        raise ToolError(ocr.INSTALL_HINT)
    box = None if cell is None and region is None else region_or_error(cell, region)
    try:
        langs = ocr.normalize_languages(languages)
    except ocr.OCRError as exc:
        raise ToolError(str(exc)) from exc

    if not ocr.is_loaded(langs):
        await ctx.info(
            f"Loading OCR models for {', '.join(langs)}. The first use downloads them, which can take a while."
        )
    try:
        data = await load_image(image)
    except Exception as exc:
        raise ToolError(f"Could not load the image: {exc}") from exc

    await ctx.info("Reading text in the image" if box is None else "Reading text in the selected region")
    try:
        result = await asyncio.to_thread(ocr.read_text, data, langs, min_confidence, box)
    except ValueError as exc:  # OCRError, or an image that cannot be decoded
        raise ToolError(str(exc)) from exc

    await ctx.info(f"Found {result.total_blocks} text blocks")
    out = asdict(result)
    if box is None:
        del out["region"], out["source_pixels"]
    if cell is not None:
        out["cell"] = cell.strip().upper()
    if not result.blocks:
        out["note"] = (
            "No text found above the confidence threshold. Try a lower min_confidence, or other "
            "languages if the text uses a non-Latin script."
            + ("" if box is not None else " Small or distant text may be read better with cell or region.")
        )
    return out
