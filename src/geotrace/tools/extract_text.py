import asyncio
from dataclasses import asdict
from typing import Annotated

from mcp.server.mcpserver import Context
from mcp.server.mcpserver.exceptions import ToolError
from pydantic import Field

from geotrace.core import ocr
from geotrace.core.images import load_image
from geotrace.server import mcp


@mcp.tool(
    name="extract_text",
    description="Detects and reads text in an image with OCR (EasyOCR): signs, shop names, street "
    "names, bus stop names, posters, house numbers. Returns blocks sorted with the largest, most "
    "confident text first (at most 50). Each block has text, confidence (0-1), bbox (x1, y1, x2, y2 "
    "in image pixels) and relative_size (share of the image area; big text is usually a sign). "
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
) -> dict:
    if not ocr.is_available():
        raise ToolError(ocr.INSTALL_HINT)
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

    await ctx.info("Reading text in the image")
    try:
        result = await asyncio.to_thread(ocr.read_text, data, langs, min_confidence)
    except ValueError as exc:  # OCRError, or an image that cannot be decoded
        raise ToolError(str(exc)) from exc

    await ctx.info(f"Found {result.total_blocks} text blocks")
    out = asdict(result)
    if not result.blocks:
        out["note"] = (
            "No text found above the confidence threshold. Try a lower min_confidence, or other "
            "languages if the text uses a non-Latin script."
        )
    return out
