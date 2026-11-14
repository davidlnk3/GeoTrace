"""Semantic lookup of a curated table of clothing and objects with known dates.

The table (geotrace/data/fashion_clues.md) lists about 170 items with the year each one
appears, the years it peaks, and how strong a dating clue it is. Rows are matched by
meaning rather than by string, so "clamshell phone" finds "Motorola RAZR flip phone":
every row is embedded once with the EVA02-CLIP model that core.time_objects already
loads, and a query is matched by cosine similarity.

Two independent queries are offered. match_objects compares the names the model says it
saw with the rows: a confident match gives a "not before" year for the photo.
suggest_from_image compares the image itself with the rows, which only ever yields
hints to check by looking at the image, never evidence.

The table is a curated, approximate reference. Fashion is regional and cyclical, and the
rows for 1970-2019 were compiled without per-row sources.
"""
import hashlib
import re
from dataclasses import asdict, dataclass
from importlib import resources
from pathlib import Path
from typing import Any

from geotrace.core.time_objects import (
    MODEL_NAME,
    PRETRAINED,
    _embed_text,
    _load,
)

TABLE_FILE = "fashion_clues.md"

ROW_TEMPLATE = "A photo of {item}."
BATCH = 64

# Calibrated against the real table with EVA02-L-14-336. CLIP text-text similarity is high even
# for unrelated pairs: a handful of generic rows ("Quiet luxury", "Velvet", "Hot pants") are the
# nearest row for any query, so "cathedral" scores 0.757 and "bicycle" 0.729 against them. Real
# matches have to clear that noise floor, hence a threshold far above zero: 0.78 drops every
# unrelated probe, while keeping "airpods" (0.952), "crocs" (1.000), "fidget spinner" (0.918),
# "skinny jeans" (0.908), "velour tracksuit" (0.833) and "football shirt" (0.831). The cost is
# that a terse query can fall just short, as "flip phone" does at 0.735 against "Motorola RAZR
# flip phone"; returning nothing is the safer error for a tool that dates evidence.
TEXT_MATCH_THRESHOLD = 0.78
# Image-text similarity lives in a much narrower band: over six probe photos every row scored
# between -0.02 and 0.205, with the 99th percentile at 0.164. 0.18 sits above that noise, so
# photos with no clothing in them (a coffee cup, gravel, a camera) yield no hints at all.
IMAGE_SUGGESTION_THRESHOLD = 0.18
# How far the best row must beat the runner-up for a text match to count as confident. Unrelated
# queries land on a cluster of near-equal generic rows, so a tie is itself a sign of a bad match.
TEXT_MATCH_MARGIN = 0.02

_DECADE_OFFSET = {"early": 0, "mid": 4, "late": 7}
_DECADE_RE = re.compile(r"(early|mid|late)[-\s]+(\d{3})0s", re.IGNORECASE)
_PLAIN_DECADE_RE = re.compile(r"\b(\d{3})0s\b")
_YEAR_RE = re.compile(r"\b(1[89]\d{2}|20\d{2})\b")
_QUOTE_RE = re.compile(r'"([^"]+)"')
_STRENGTHS = ("medium-strong", "strong", "medium", "weak")

# numpy is imported lazily, as everywhere else in core, so the arrays stay untyped here.
_MEMORY_CACHE: dict[str, Any] = {}


class ClueTableError(ValueError):
    """The dating-clues table could not be parsed."""


@dataclass(frozen=True)
class ClueRow:
    """One row of the table: the original text plus the years read out of it."""

    item: str
    appears: str                    # original text, e.g. "Late 2004"
    peak: str                       # original text, e.g. "2005-2007"
    clue_type: str                  # original text, e.g. 'Strong ("not before late 2004")'
    appears_year: int | None
    peak_from: int | None
    peak_to: int | None
    strength: str                   # strong | medium-strong | medium | weak
    not_before_year: int | None


@dataclass(frozen=True)
class ClueMatch:
    """One row returned by a search, with its similarity to the query."""

    row: ClueRow
    similarity: float
    confident: bool

    def to_dict(self) -> dict:
        out = asdict(self.row)
        out.pop("appears_year")
        out.pop("peak_from")
        out.pop("peak_to")
        out.pop("strength")
        out["similarity"] = self.similarity
        out["confident"] = self.confident
        return out


def _fail(line_no: int, line: str, problem: str) -> ClueTableError:
    return ClueTableError(f"line {line_no}: {problem}: {line!r}")


def _single_year(text: str) -> int | None:
    """The one year a phrase refers to, or None if it names no year."""
    m = _DECADE_RE.search(text)
    if m:
        return int(m.group(2)) * 10 + _DECADE_OFFSET[m.group(1).lower()]
    m = _PLAIN_DECADE_RE.search(text)
    if m:
        return int(m.group(1)) * 10
    m = _YEAR_RE.search(text)
    return int(m.group(1)) if m else None


def _parse_appears(text: str, line_no: int, line: str) -> int | None:
    if text.lower().startswith("(already existed"):
        return None
    year = _single_year(text)
    if year is None:
        raise _fail(line_no, line, f"cannot read a year from the appears column {text!r}")
    return year


def _parse_peak(text: str, line_no: int, line: str) -> tuple[int | None, int | None]:
    low = text.lower()
    if low.startswith("(ongoing"):
        return None, None
    if "–" in text:                                  # en dash range, e.g. "2016-2017"
        left, right = text.split("–", 1)
        start, end = _single_year(left), _single_year(right)
        if start is None or end is None:
            raise _fail(line_no, line, f"cannot read the peak range {text!r}")
        return start, end
    if "onwards" in low:
        start = _single_year(text)
        if start is None:
            raise _fail(line_no, line, f"cannot read the start of the peak {text!r}")
        return start, None
    m = _PLAIN_DECADE_RE.search(text)
    if m:
        return int(m.group(1)) * 10, int(m.group(1)) * 10 + 9
    if len(set(_YEAR_RE.findall(text))) > 1:
        # Two years with no en dash between them: an unknown range spelling, not a single year.
        raise _fail(line_no, line, f"ambiguous peak {text!r}")
    year = _single_year(text)
    if year is None:
        raise _fail(line_no, line, f"cannot read a year from the peak column {text!r}")
    return year, year


def _parse_strength(text: str, line_no: int, line: str) -> str:
    low = text.lower()
    for strength in _STRENGTHS:
        if low.startswith(strength):
            return strength
    raise _fail(line_no, line, f"unknown clue type {text!r}")


def _parse_not_before(clue_type: str, strength: str, line_no: int, line: str) -> int | None:
    """The "not before" year a strong or medium clue states, if it states one."""
    if strength == "weak":
        return None
    quoted = _QUOTE_RE.search(clue_type)
    if quoted is None:
        return None                                       # e.g. "Strong (2020-2022 in Europe)"
    phrase = quoted.group(1).strip().lower()
    if phrase.startswith("around"):
        return None                                       # an era, not a lower bound
    year = _single_year(phrase)
    if year is None:
        raise _fail(line_no, line, f"cannot read a year from the clue type {clue_type!r}")
    return year


def parse_table(text: str) -> list[ClueRow]:
    """Parse the Markdown table. Raises ClueTableError naming the line of any bad row."""
    rows: list[ClueRow] = []
    for line_no, line in enumerate(text.splitlines(), start=1):
        stripped = line.strip()
        if not stripped.startswith("|"):
            continue
        cells = [c.strip() for c in stripped.strip("|").split("|")]
        if set("".join(cells)) <= set("-: "):
            continue                                      # the |---|---| separator
        if cells and cells[0].lower().startswith("clothing item"):
            continue                                      # the header
        if len(cells) != 4:
            raise _fail(line_no, line, f"expected 4 columns, found {len(cells)}")
        item, appears, peak, clue_type = cells
        if not item:
            raise _fail(line_no, line, "the item name is empty")
        strength = _parse_strength(clue_type, line_no, line)
        peak_from, peak_to = _parse_peak(peak, line_no, line)
        rows.append(ClueRow(
            item=item,
            appears=appears,
            peak=peak,
            clue_type=clue_type,
            appears_year=_parse_appears(appears, line_no, line),
            peak_from=peak_from,
            peak_to=peak_to,
            strength=strength,
            not_before_year=_parse_not_before(clue_type, strength, line_no, line),
        ))
    if not rows:
        raise ClueTableError("the dating-clues table has no rows")
    return rows


def table_path() -> Path:
    """The table file, read through importlib so it works installed as well as from the repo."""
    packaged = resources.files("geotrace") / "data" / TABLE_FILE
    if not packaged.is_file():
        raise FileNotFoundError(f"the dating-clues table is missing from the package: "
                                f"geotrace/data/{TABLE_FILE}")
    return Path(str(packaged))


def table_text() -> str:
    return table_path().read_text(encoding="utf-8")


def cache_key(text: str, model_name: str = MODEL_NAME, pretrained: str = PRETRAINED) -> str:
    """A digest of the table and the model, so either one changing invalidates the cache."""
    h = hashlib.sha256()
    h.update(text.encode("utf-8"))
    h.update(b"\0")
    h.update(model_name.encode("utf-8"))
    h.update(b"\0")
    h.update(pretrained.encode("utf-8"))
    return h.hexdigest()[:16]


def _cache_file(key: str) -> Path | None:
    """Where the row embeddings live on disk, or None without platformdirs."""
    try:
        from platformdirs import user_cache_dir
    except ImportError:
        # platformdirs is not a dependency of the geoclip extra, so fall back to memory only.
        return None
    return Path(user_cache_dir("geotrace")) / f"dating_clues_{key}.npy"


def row_embeddings_cached(text: str | None = None) -> bool:
    """Whether the row embeddings are ready, so the caller can warn before a slow first run."""
    key = cache_key(table_text() if text is None else text)
    if key in _MEMORY_CACHE:
        return True
    path = _cache_file(key)
    return path is not None and path.is_file()


def row_embeddings(rows: list[ClueRow], text: str) -> Any:
    """The normalised embedding of every row, from the cache when it is there."""
    import numpy as np

    key = cache_key(text)
    cached = _MEMORY_CACHE.get(key)
    if cached is not None:
        return cached

    path = _cache_file(key)
    if path is not None and path.is_file():
        emb = np.load(path)
        if emb.shape[0] == len(rows):
            _MEMORY_CACHE[key] = emb
            return emb

    model, _, tokenizer = _load()
    chunks = []
    for start in range(0, len(rows), BATCH):
        texts = [ROW_TEMPLATE.format(item=r.item) for r in rows[start:start + BATCH]]
        chunks.append(_embed_text(model, tokenizer, texts).detach().cpu().numpy())
    emb = np.concatenate(chunks).astype(np.float32)

    if path is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.save(path, emb)
    _MEMORY_CACHE[key] = emb
    return emb


def _as_array(embedding) -> Any:
    """A float32 numpy view of a torch tensor or an array."""
    import numpy as np

    if hasattr(embedding, "detach"):
        embedding = embedding.detach().cpu().numpy()
    return np.asarray(embedding, dtype=np.float32)


def _table() -> tuple[list[ClueRow], Any]:
    text = table_text()
    rows = parse_table(text)
    return rows, row_embeddings(rows, text)


def match_objects(names: list[str], top_k: int = 3) -> dict[str, list[ClueMatch]]:
    """The table rows that mean the same as each name, best first.

    Only rows above TEXT_MATCH_THRESHOLD are returned, so "no match" comes back as an empty
    list rather than the best weak guess. Only the top row can be `confident`, and only when
    it beats the runner-up by TEXT_MATCH_MARGIN: a tie between rows means the query was
    ambiguous, so it must not be read as a dated fact.
    """
    import numpy as np

    rows, emb = _table()
    model, _, tokenizer = _load()
    queries = _embed_text(model, tokenizer, [ROW_TEMPLATE.format(item=n) for n in names])
    sims = _as_array(queries).reshape(len(names), -1) @ emb.T

    out: dict[str, list[ClueMatch]] = {}
    for i, name in enumerate(names):
        order = np.argsort(-sims[i])
        best = float(sims[i][order[0]])
        runner_up = float(sims[i][order[1]]) if len(order) > 1 else -1.0
        confident = best - runner_up >= TEXT_MATCH_MARGIN
        matches = []
        for rank, j in enumerate(order[:top_k]):
            score = float(sims[i][j])
            if score < TEXT_MATCH_THRESHOLD:
                break
            matches.append(ClueMatch(rows[j], round(score, 4), rank == 0 and confident))
        out[name] = matches
    return out


def suggest_from_image(image_embedding, top_k: int = 5) -> list[ClueMatch]:
    """Table items that may be visible in the photo. Unverified hints, never evidence."""
    import numpy as np

    rows, emb = _table()
    sims = emb @ _as_array(image_embedding).reshape(-1)
    order = np.argsort(-sims)
    return [
        ClueMatch(rows[j], round(float(sims[j]), 4), False)
        for j in order[:top_k]
        if float(sims[j]) >= IMAGE_SUGGESTION_THRESHOLD
    ]


def lookup(names: list[str], image_embedding) -> dict:
    """The whole dating-clues signal for one call: text matches, image hints and the bound."""
    matches = match_objects(names)
    confident_years = [
        m.row.not_before_year
        for ms in matches.values()
        for m in ms
        if m.confident and m.row.not_before_year is not None
    ]
    return {
        "matches": {name: [m.to_dict() for m in ms] for name, ms in matches.items()},
        # Unverified: what the image embedding resembles, to check by looking at the image.
        "image_suggestions": [m.to_dict() for m in suggest_from_image(image_embedding)],
        # Only confident text matches, never image suggestions, can bound the date.
        "table_not_before_year": max(confident_years) if confident_years else None,
    }
