"""Semantic lookup of a curated table of clothing and objects with known dates.

The table (geotrace/data/fashion_clues.md) lists about 170 items with the year each one
appears, the years it peaks, and how strong a dating clue it is. Rows are matched by
meaning rather than by string, so "clamshell phone" finds "Motorola RAZR flip phone":
every row is embedded once with the EVA02-CLIP model that core.time_objects already
loads, and a query is matched by cosine similarity.

CLIP text embeddings are hubs: a few rows ("Quiet luxury", "Velvet") are the nearest
neighbour of almost any query, so a plain cosine ranks "cathedral" above a real match.
Two corrections are applied. The row mean is subtracted from every row and every query
before renormalising, which removes the shared direction; and each row carries the mean
and standard deviation of its similarity to a fixed set of unrelated background queries,
so a match is scored as z = (similarity - row mean) / row standard deviation. A hub row
has a high background mean and is penalised accordingly.

Two independent queries are offered. match_objects compares the names the model says it
saw with the rows: a match that also shares a word with the row is marked confident and
can give a "not before" year for the photo. suggest_from_image compares the image itself
with the rows, which only ever yields hints to check by looking at the image.

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

# Unrelated queries that stand for "an ordinary photographic subject". Every row's similarity
# to these gives it a baseline, which is what turns a raw cosine into a z-score. They are part
# of the cache key, so editing this list recomputes the statistics.
BACKGROUND_QUERIES = (
    "a cathedral", "the Eiffel Tower", "a suburban house", "a shopping mall",
    "a railway station", "a lighthouse",
    "a dog", "a horse", "a seagull", "a cat sleeping", "a cow in a field",
    "a bicycle", "a city bus", "a parked car", "a cargo ship", "an aeroplane", "a tractor",
    "a mountain range", "a pine forest", "a sandy beach", "a river at dusk",
    "a field of wheat", "storm clouds",
    "a bowl of soup", "a loaf of bread", "a plate of pasta", "an apple",
    "a cup of coffee", "a birthday cake",
    "a traffic light", "a fire hydrant", "a park bench", "a brick wall",
)

# Calibrated on the real table with EVA02-L-14-336, scoring on z after centring. Measured on
# 25 held-out unrelated queries (none of them in BACKGROUND_QUERIES) and 15 real object names:
# the unrelated queries peak at z = 9.9 and the genuine matches start at z = 5.9, so the two
# still overlap and z alone cannot decide. z therefore only decides what is worth returning,
# and a shared word decides what is trustworthy (see _shares_words).
Z_MATCH_THRESHOLD = 5.0
# Image-text similarity is a different modality on a different scale, so the text background
# statistics do not transfer and centring measurably did not help: over six probe photos the
# centred maximum for gravel (+0.122) matched a real street scene (+0.123). The image path
# therefore stays on the plain cosine against the uncentred rows, where every row scored
# between -0.02 and 0.205 with the 99th percentile at 0.164.
IMAGE_SUGGESTION_THRESHOLD = 0.18

# Filler words that must never, on their own, make a match look confirmed.
_STOP_WORDS = frozenset({
    "and", "the", "with", "worn", "style", "first", "wave", "dominant", "cut",
    "everywhere", "its", "for", "as", "in", "on", "to", "or", "a", "an", "of",
})

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
    """One row returned by a search, with how well it scored.

    `similarity` is the plain cosine, kept because it is the number a human recognises.
    `z` is what the thresholds act on: how far above its own background baseline the row
    scored, in standard deviations. It is None for image suggestions, which are not
    scored against the text background. `confident` means the match is safe to date with.
    """

    row: ClueRow
    similarity: float
    z: float | None
    confident: bool

    def to_dict(self) -> dict:
        out = asdict(self.row)
        for numeric in ("appears_year", "peak_from", "peak_to", "strength"):
            out.pop(numeric)
        out["similarity"] = self.similarity
        out["z"] = self.z
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


def cache_key(text: str, model_name: str = MODEL_NAME, pretrained: str = PRETRAINED,
              background: tuple[str, ...] = BACKGROUND_QUERIES) -> str:
    """A digest of everything the cached arrays depend on: table, model and background set."""
    h = hashlib.sha256()
    for part in (text, model_name, pretrained, "\n".join(background)):
        h.update(part.encode("utf-8"))
        h.update(b"\0")
    return h.hexdigest()[:16]


def _cache_file(key: str) -> Path | None:
    """Where the row embeddings live on disk, or None without platformdirs."""
    try:
        from platformdirs import user_cache_dir
    except ImportError:
        # platformdirs is not a dependency of the geoclip extra, so fall back to memory only.
        return None
    return Path(user_cache_dir("geotrace")) / f"dating_clues_{key}.npz"


def row_embeddings_cached(text: str | None = None) -> bool:
    """Whether the row embeddings are ready, so the caller can warn before a slow first run."""
    key = cache_key(table_text() if text is None else text)
    if key in _MEMORY_CACHE:
        return True
    path = _cache_file(key)
    return path is not None and path.is_file()


def _unitise(vectors) -> Any:
    """Scale each row vector to unit length, so a dot product is a cosine."""
    import numpy as np

    norms = np.linalg.norm(vectors, axis=-1, keepdims=True)
    return vectors / np.maximum(norms, 1e-12)


def _embed_rows(texts: list[str]) -> Any:
    """Embed `texts` through the row template, in batches, as float32."""
    import numpy as np

    model, _, tokenizer = _load()
    chunks = []
    for start in range(0, len(texts), BATCH):
        batch = [ROW_TEMPLATE.format(item=t) for t in texts[start:start + BATCH]]
        chunks.append(_as_array(_embed_text(model, tokenizer, batch)))
    return np.concatenate(chunks).astype(np.float32)


def build_index(rows: list[ClueRow]) -> dict:
    """Embed the rows and measure each one against the background queries.

    Returns the raw embeddings (for the image path), the centred ones and the row mean
    (for the text path), and each row's background mean and standard deviation.
    """
    raw = _embed_rows([r.item for r in rows])
    mean_row = raw.mean(axis=0)
    centred = _unitise(raw - mean_row)

    background = _unitise(_embed_rows(list(BACKGROUND_QUERIES)) - mean_row)
    background_sims = centred @ background.T          # (rows, background queries)
    return {
        "raw": raw,
        "centred": centred,
        "mean_row": mean_row,
        "background_mean": background_sims.mean(axis=1),
        # Guard the divisor: a row with no spread would otherwise give an infinite z.
        "background_std": background_sims.std(axis=1).clip(min=1e-6),
    }


def row_embeddings(rows: list[ClueRow], text: str) -> dict:
    """The index for these rows, from the cache when it is there."""
    import numpy as np

    key = cache_key(text)
    cached = _MEMORY_CACHE.get(key)
    if cached is not None:
        return cached

    path = _cache_file(key)
    if path is not None and path.is_file():
        with np.load(path) as stored:
            index = {k: stored[k] for k in stored.files}
        if index["raw"].shape[0] == len(rows):
            _MEMORY_CACHE[key] = index
            return index

    index = build_index(rows)
    if path is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(path, **index)
    _MEMORY_CACHE[key] = index
    return index


def _as_array(embedding) -> Any:
    """A float32 numpy view of a torch tensor or an array."""
    import numpy as np

    if hasattr(embedding, "detach"):
        embedding = embedding.detach().cpu().numpy()
    return np.asarray(embedding, dtype=np.float32)


def _words(text: str) -> set[str]:
    """The content words of a name, lowercased and crudely de-pluralised."""
    out = set()
    for word in "".join(c.lower() if c.isalnum() else " " for c in text).split():
        if len(word) < 3 or word in _STOP_WORDS:
            continue
        out.add(word[:-1] if word.endswith("s") and len(word) > 4 else word)
    return out


def _shares_words(query: str, item: str) -> bool:
    """Whether a query and a row name have a content word in common.

    This is what separates a real match from a plausible-looking one. Centring and the
    z-score fix the ranking, but they cannot tell "a crocodile" (z 9.9 against "Classic
    Crocs") from "labubu" (z 5.9 against the row that names it); a shared word can.
    """
    return bool(_words(query) & _words(item))


def _table() -> tuple[list[ClueRow], dict]:
    text = table_text()
    rows = parse_table(text)
    return rows, row_embeddings(rows, text)


def match_objects(names: list[str], top_k: int = 3) -> dict[str, list[ClueMatch]]:
    """The table rows that mean the same as each name, best first.

    Rows are ranked by z, not by raw similarity, so hub rows do not win by default. Only
    rows at or above Z_MATCH_THRESHOLD come back, so "no match" is an empty list rather
    than the best weak guess. A row is `confident` only if it also shares a word with the
    name: a purely semantic neighbour is returned as a candidate to check, never as a fact.
    """
    import numpy as np

    rows, index = _table()
    model, _, tokenizer = _load()
    queries = _as_array(_embed_text(
        model, tokenizer, [ROW_TEMPLATE.format(item=n) for n in names]))

    cosine = queries @ index["raw"].T
    centred = _unitise(queries - index["mean_row"])
    z = (centred @ index["centred"].T - index["background_mean"]) / index["background_std"]

    out: dict[str, list[ClueMatch]] = {}
    for i, name in enumerate(names):
        matches = []
        for j in np.argsort(-z[i])[:top_k]:
            if z[i][j] < Z_MATCH_THRESHOLD:
                break
            matches.append(ClueMatch(
                row=rows[j],
                similarity=round(float(cosine[i][j]), 4),
                z=round(float(z[i][j]), 2),
                confident=_shares_words(name, rows[j].item),
            ))
        out[name] = matches
    return out


def suggest_from_image(image_embedding, top_k: int = 5) -> list[ClueMatch]:
    """Table items that may be visible in the photo. Unverified hints, never evidence.

    Scored on the plain cosine against the uncentred rows: the text background statistics
    belong to another modality, so there is no z to report here.
    """
    import numpy as np

    rows, index = _table()
    sims = index["raw"] @ _unitise(_as_array(image_embedding).reshape(-1))
    return [
        ClueMatch(row=rows[j], similarity=round(float(sims[j]), 4), z=None, confident=False)
        for j in np.argsort(-sims)[:top_k]
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
