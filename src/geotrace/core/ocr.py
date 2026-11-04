"""Text detection and recognition (OCR) in images.

The OCR engine sits behind the small Backend protocol, so another engine (for example
PaddleOCR) can be added without changing the tool. Only EasyOCR is implemented.

Language rules follow EasyOCR 1.7.2 (easyocr/easyocr.py, Reader.__init__ and
easyocr/config.py): Latin-script languages can be combined freely; the other scripts each
have their own model and can only be combined with English (and, for Arabic, Bengali,
Cyrillic and Devanagari, with other languages of the same script). The rules are copied
here so an invalid combination fails before any model is loaded or downloaded.
"""

from __future__ import annotations

import contextlib
import importlib.util
import math
import sys
import threading
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Protocol

from PIL import Image

from geotrace.core.images import open_rgb

# --- Configuration -----------------------------------------------------------------------

DEFAULT_LANGUAGES = ["en", "es", "fr", "de", "it", "pt", "nl"]
DEFAULT_MIN_CONFIDENCE = 0.3
MAX_BLOCKS = 50
# Only larger images are downscaled: small text needs resolution.
MAX_SIDE = 3000
# EasyOCR also tries each detected box rotated by these angles and keeps the best reading,
# which helps with vertical or upside-down text.
ROTATION_INFO = [90, 180, 270]

INSTALL_HINT = (
    "OCR needs the optional 'ocr' extra, which is not installed. "
    "Install it with `uv sync --extra ocr` (or `pip install -e \".[ocr]\"`) and restart the server."
)

# --- EasyOCR language groups (easyocr/config.py, v1.7.2) ---------------------------------

LATIN = [
    "af", "az", "bs", "cs", "cy", "da", "de", "en", "es", "et", "fr", "ga", "hr", "hu", "id",
    "is", "it", "ku", "la", "lt", "lv", "mi", "ms", "mt", "nl", "no", "oc", "pi", "pl", "pt",
    "ro", "rs_latin", "sk", "sl", "sq", "sv", "sw", "tl", "tr", "uz", "vi",
]
ARABIC = ["ar", "fa", "ug", "ur"]
BENGALI = ["bn", "as", "mni"]
CYRILLIC = [
    "ru", "rs_cyrillic", "be", "bg", "uk", "mn", "abq", "ady", "kbd", "ava", "dar", "inh",
    "che", "lbe", "lez", "tab", "tjk",
]
DEVANAGARI = [
    "hi", "mr", "ne", "bh", "mai", "ang", "bho", "mah", "sck", "new", "gom", "sa", "bgc",
]
OTHER = ["th", "ch_sim", "ch_tra", "ja", "ko", "ta", "te", "kn"]
ALL_LANGUAGES = set(LATIN + ARABIC + BENGALI + CYRILLIC + DEVANAGARI + OTHER)

# EasyOCR picks the model with an if/elif chain; the first matching rule wins and then every
# requested language must be in that rule's allowed set. Same order as Reader.__init__.
_SINGLE_SCRIPT = [
    ("th", "Thai"), ("ch_tra", "Traditional Chinese"), ("ch_sim", "Simplified Chinese"),
    ("ja", "Japanese"), ("ko", "Korean"), ("ta", "Tamil"), ("te", "Telugu"), ("kn", "Kannada"),
]
_SCRIPT_GROUPS = [
    ("Bengali", BENGALI), ("Arabic", ARABIC), ("Devanagari", DEVANAGARI), ("Cyrillic", CYRILLIC),
]


class OCRError(ValueError):
    """Invalid input or OCR failure. The message is safe to show the model."""


class OCRUnavailableError(OCRError):
    """The OCR extra is not installed."""


def normalize_languages(languages: Sequence[str] | None) -> list[str]:
    """Return the language list to use, validated against EasyOCR's combination rules.

    None or an empty list gives DEFAULT_LANGUAGES. Codes are lower-cased and deduplicated.
    Raises OCRError for unknown codes or combinations EasyOCR cannot load in one reader.
    """
    if not languages:
        return list(DEFAULT_LANGUAGES)

    langs: list[str] = []
    for code in languages:
        c = code.strip().lower()
        if c and c not in langs:
            langs.append(c)
    if not langs:
        return list(DEFAULT_LANGUAGES)

    unknown = [c for c in langs if c not in ALL_LANGUAGES]
    if unknown:
        raise OCRError(
            f"Unsupported language code(s): {', '.join(unknown)}. Use EasyOCR codes, for example "
            "en, es, fr, de, ru, ar, ja, ko, ch_sim, ch_tra, hi, th "
            "(see https://www.jaided.ai/easyocr/)."
        )

    script, allowed = _model_for(langs)
    rejected = [c for c in langs if c not in allowed]
    if rejected:
        keep = [c for c in langs if c in allowed]
        raise OCRError(
            f"{script} ({', '.join(keep)}) cannot be combined with {', '.join(rejected)} in one OCR call: "
            f"EasyOCR only combines {script} with "
            f"{'English' if len(allowed) == 2 else 'English and other ' + script + ' languages'}. "
            f"Split them into separate calls, for example languages={keep} and then "
            f"languages={rejected}."
        )
    return langs


def _model_for(langs: list[str]) -> tuple[str, set[str]]:
    """Which EasyOCR model the list selects, and which codes that model accepts."""
    for code, script in _SINGLE_SCRIPT:
        if code in langs:
            return script, {code, "en"}
    for script, group in _SCRIPT_GROUPS:
        if set(langs) & set(group):
            return script, set(group) | {"en"}
    return "Latin", set(LATIN)


# --- Result types ------------------------------------------------------------------------

@dataclass
class TextBlock:
    text: str
    confidence: float
    bbox: tuple[int, int, int, int]  # x1, y1, x2, y2 in image pixels
    relative_size: float  # block area / image area: bigger text is usually a sign


@dataclass
class OCRResult:
    image_size: tuple[int, int]
    languages: list[str]
    blocks: list[TextBlock] = field(default_factory=list)  # best first, at most MAX_BLOCKS
    total_blocks: int = 0
    truncated: bool = False


# One raw detection: 4 corner points [x, y], text, confidence.
Detection = tuple[Sequence[Sequence[float]], str, float]


# --- Backends ----------------------------------------------------------------------------

class Backend(Protocol):
    def read(self, image: Image.Image) -> list[Detection]: ...


def is_available() -> bool:
    """True if the EasyOCR package can be imported."""
    return importlib.util.find_spec("easyocr") is not None


class EasyOCRBackend:
    """EasyOCR reader for one language set. Creating it loads (and on first use downloads) the models."""

    def __init__(self, languages: list[str]):
        try:
            import easyocr  # pyright: ignore[reportMissingImports]  # optional "ocr" extra
        except ImportError as exc:
            raise OCRUnavailableError(INSTALL_HINT) from exc
        # EasyOCR prints progress to stdout, which would corrupt the MCP stdio stream.
        with contextlib.redirect_stdout(sys.stderr):
            try:
                self._reader = easyocr.Reader(languages, gpu=True, verbose=False)
            except ValueError as exc:
                raise OCRError(f"EasyOCR rejected languages {languages}: {exc}") from exc

    def read(self, image: Image.Image) -> list[Detection]:
        import numpy as np

        with contextlib.redirect_stdout(sys.stderr):
            raw = self._reader.readtext(np.asarray(image), detail=1, rotation_info=ROTATION_INFO)
        return [(box, str(text), float(conf)) for box, text, conf in raw]


BackendFactory = Callable[[list[str]], Backend]

_backend_factory: BackendFactory = EasyOCRBackend
_readers: dict[tuple[str, ...], Backend] = {}
_readers_lock = threading.Lock()


def _key(languages: list[str]) -> tuple[str, ...]:
    return tuple(sorted(languages))


def is_loaded(languages: list[str]) -> bool:
    """True if a reader for this language set is already in memory."""
    return _key(languages) in _readers


def get_reader(languages: list[str]) -> Backend:
    """Return the cached reader for this language set, creating it on first use."""
    key = _key(languages)
    with _readers_lock:
        reader = _readers.get(key)
        if reader is None:
            reader = _backend_factory(list(key))
            _readers[key] = reader
        return reader


# --- Post-processing ---------------------------------------------------------------------

def _to_bbox(points: Sequence[Sequence[float]], width: int, height: int) -> tuple[int, int, int, int]:
    xs = [float(p[0]) for p in points]
    ys = [float(p[1]) for p in points]
    x1 = max(0, min(width, math.floor(min(xs))))
    y1 = max(0, min(height, math.floor(min(ys))))
    x2 = max(0, min(width, math.ceil(max(xs))))
    y2 = max(0, min(height, math.ceil(max(ys))))
    return x1, y1, x2, y2


def _same_line(left: TextBlock, right: TextBlock) -> bool:
    """True if right continues left on the same text line: similar height, same baseline, small gap."""
    hl = left.bbox[3] - left.bbox[1]
    hr = right.bbox[3] - right.bbox[1]
    if hl <= 0 or hr <= 0 or min(hl, hr) / max(hl, hr) < 0.7:
        return False
    h = (hl + hr) / 2
    if abs(left.bbox[3] - right.bbox[3]) > 0.3 * h:  # baseline (bottom edge)
        return False
    gap = right.bbox[0] - left.bbox[2]
    return -0.2 * h <= gap <= 1.0 * h


def group_lines(blocks: list[TextBlock]) -> list[list[TextBlock]]:
    """Group blocks that read left to right on the same line, such as "VIA" + "ROMA"."""
    lines: list[list[TextBlock]] = []
    for block in sorted(blocks, key=lambda b: (b.bbox[0], b.bbox[1])):
        candidates = [line for line in lines if _same_line(line[-1], block)]
        if candidates:
            best = min(candidates, key=lambda line: block.bbox[0] - line[-1].bbox[2])
            best.append(block)
        else:
            lines.append([block])
    return lines


def _merge(line: list[TextBlock], image_area: int) -> TextBlock:
    if len(line) == 1:
        return line[0]
    x1 = min(b.bbox[0] for b in line)
    y1 = min(b.bbox[1] for b in line)
    x2 = max(b.bbox[2] for b in line)
    y2 = max(b.bbox[3] for b in line)
    return TextBlock(
        text=" ".join(b.text for b in line),
        # A merged line is only as reliable as its weakest part.
        confidence=min(b.confidence for b in line),
        bbox=(x1, y1, x2, y2),
        relative_size=(x2 - x1) * (y2 - y1) / image_area,
    )


def score(block: TextBlock) -> float:
    """Ranking score: confident, large text first. sqrt keeps size from swamping confidence."""
    return block.confidence * math.sqrt(block.relative_size)


def process_detections(
    detections: Sequence[Detection],
    image_size: tuple[int, int],
    min_confidence: float = DEFAULT_MIN_CONFIDENCE,
    max_blocks: int = MAX_BLOCKS,
) -> tuple[list[TextBlock], int]:
    """Filter, group into lines, sort and truncate raw detections. Returns (blocks, total_blocks)."""
    width, height = image_size
    area = max(1, width * height)
    blocks: list[TextBlock] = []
    for points, text, conf in detections:
        text = " ".join(text.split())
        if not text or conf < min_confidence:
            continue
        bbox = _to_bbox(points, width, height)
        rel = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1]) / area
        blocks.append(TextBlock(text=text, confidence=round(float(conf), 4), bbox=bbox, relative_size=rel))

    merged = [_merge(line, area) for line in group_lines(blocks)]
    merged.sort(key=score, reverse=True)
    for b in merged:
        b.relative_size = round(b.relative_size, 6)
    return merged[:max_blocks], len(merged)


def read_text(
    image_bytes: bytes,
    languages: Sequence[str] | None = None,
    min_confidence: float = DEFAULT_MIN_CONFIDENCE,
) -> OCRResult:
    """Run OCR on an image. Synchronous and slow: call it from a worker thread."""
    if not 0 <= min_confidence <= 1:
        raise OCRError("min_confidence must be between 0 and 1")
    langs = normalize_languages(languages)
    image = open_rgb(image_bytes, max_side=MAX_SIDE)
    reader = get_reader(langs)
    detections = reader.read(image)
    blocks, total = process_detections(detections, image.size, min_confidence)
    return OCRResult(
        image_size=image.size,
        languages=langs,
        blocks=blocks,
        total_blocks=total,
        truncated=total > len(blocks),
    )
