"""Search OpenStreetMap for features whose name matches a text (for example one read by OCR).

The text is untrusted: it is escaped so it is matched literally by the Overpass regex
engine (POSIX extended regular expressions) and cannot break out of the QL string.
Matching is case-insensitive, accent-insensitive for common Latin letters and on collapsed
whitespace. It is a substring match: "Farmacia Roma" finds "Antica Farmacia Roma".
"""

from __future__ import annotations

import math
import re
import unicodedata
from dataclasses import dataclass

from geotrace.core import overpass

MIN_NAME_LENGTH = 2
MAX_NAME_LENGTH = 100
MAX_TAG_FILTERS = 5
MAX_RESULTS = 25
# Upper bound on elements downloaded; the nearest MAX_RESULTS of these are returned.
MAX_FETCH = 200
MAX_BBOX_KM2 = overpass.RARITY_MAX_BBOX_KM2["medium"]
TIMEOUT_S = overpass.DEFAULT_TIMEOUT_S

NAME_KEYS = ("name", "alt_name", "old_name", "official_name")
# name, alt_name, old_name, official_name, and their language variants such as name:en.
NAME_KEY_REGEX = r"^(name|alt_name|old_name|official_name)(:[A-Za-z_-]+)?$"

# Tags worth showing to identify a feature; others are dropped from the output.
MAIN_TAGS = (
    "amenity", "shop", "tourism", "leisure", "office", "craft", "historic", "building",
    "highway", "railway", "public_transport", "place", "landuse", "natural", "man_made",
    "brand", "operator", "addr:street", "addr:housenumber", "addr:postcode", "addr:city",
    "website", "wikidata",
)

# Characters with a special meaning in POSIX extended regular expressions.
_ERE_SPECIAL = set("\\.^$*+?()[]{}|")

# Accented forms tried for each base letter, so "Cafe" finds "Café" and the reverse.
_ACCENTS = {
    "a": "àáâãäåāăą", "c": "çćčĉċ", "d": "ďđ", "e": "èéêëēĕėęě", "g": "ğĝġģ",
    "i": "ìíîïīĭįı", "l": "łľĺļ", "n": "ñńňņ", "o": "òóôõöøōŏő", "r": "řŕŗ",
    "s": "śšşșŝ", "t": "ťţț", "u": "ùúûüūŭůűų", "y": "ýÿŷ", "z": "źżž",
}


class NamedFeatureError(ValueError):
    """Invalid input. The message is safe to show the model."""


def normalize_name(text: str) -> str:
    """NFC-normalise, drop control characters and collapse whitespace."""
    text = unicodedata.normalize("NFC", text)
    text = "".join(" " if unicodedata.category(ch).startswith("C") else ch for ch in text)
    return " ".join(text.split())


def strip_accents(text: str) -> str:
    decomposed = unicodedata.normalize("NFKD", text)
    return unicodedata.normalize("NFC", "".join(ch for ch in decomposed if not unicodedata.combining(ch)))


def escape_ere(text: str) -> str:
    """Escape text so a POSIX ERE matches it literally."""
    return "".join("\\" + ch if ch in _ERE_SPECIAL else ch for ch in text)


def _accent_insensitive(text: str) -> str:
    """Literal ERE for text where each letter with accented forms becomes an alternation.

    Alternation (e|é|...) is used instead of a bracket expression because it works byte by
    byte on UTF-8 text whatever the server's locale, and upper and lower case are both
    listed because case folding of non-ASCII letters also depends on the locale.
    """
    parts = []
    for ch in text:
        forms = _ACCENTS.get(ch.lower())
        if forms is None:
            parts.append(escape_ere(ch))
        else:
            options = ch.lower() + ch.upper() + forms + forms.upper()
            unique = "".join(dict.fromkeys(options))
            parts.append("(" + "|".join(unique) + ")")
    return "".join(parts)


def name_regex(text: str) -> str:
    """ERE matching text literally, case- and accent-insensitively, as a substring."""
    name = normalize_name(text)
    pattern = _accent_insensitive(strip_accents(name))
    # Stripping marks can change letters outside the table (e.g. Cyrillic й -> и), in which
    # case the expanded pattern no longer matches the original; add it as an alternative.
    if not re.search(pattern, name, re.IGNORECASE):
        pattern = f"{pattern}|{escape_ere(name)}"
    return pattern


def ql_string(value: str) -> str:
    """Quote a value as an Overpass QL string literal."""
    return '"' + value.replace("\\", "\\\\").replace('"', '\\"') + '"'


def validate_name(text: str) -> str:
    name = normalize_name(text)
    if len(name) < MIN_NAME_LENGTH:
        raise NamedFeatureError(f"name must have at least {MIN_NAME_LENGTH} characters")
    if len(name) > MAX_NAME_LENGTH:
        raise NamedFeatureError(
            f"name is longer than {MAX_NAME_LENGTH} characters; search a shorter, distinctive part of it"
        )
    return name


def tag_filters(tags: dict[str, str] | None) -> str:
    """Exact-match tag filters. A value of "" or "*" only requires the key to exist."""
    if not tags:
        return ""
    if len(tags) > MAX_TAG_FILTERS:
        raise NamedFeatureError(f"at most {MAX_TAG_FILTERS} tag filters are allowed")
    out = []
    for key, value in tags.items():
        key = normalize_name(key)
        value = normalize_name(value)
        if not key or len(key) > MAX_NAME_LENGTH or len(value) > MAX_NAME_LENGTH:
            raise NamedFeatureError("tag keys must be non-empty and keys and values at most 100 characters")
        if value in ("", "*"):
            out.append(f"[{ql_string(key)}]")
        else:
            out.append(f"[{ql_string(key)}={ql_string(value)}]")
    return "".join(out)


@dataclass(frozen=True)
class BBox:
    south: float
    west: float
    north: float
    east: float

    @property
    def center(self) -> tuple[float, float]:
        return (self.south + self.north) / 2, (self.west + self.east) / 2

    def ql(self) -> str:
        """Overpass order: south, west, north, east."""
        return f"{self.south},{self.west},{self.north},{self.east}"


def build_statement(name: str, bbox: BBox, tags: dict[str, str] | None = None) -> str:
    """The nwr statement selecting matching features, without header or output."""
    name = validate_name(name)
    value_re = ql_string(name_regex(name))
    key_re = ql_string(NAME_KEY_REGEX)
    return f"nwr{tag_filters(tags)}[~{key_re}~{value_re},i]({bbox.ql()});"


def build_count_query(statement: str, timeout: int = TIMEOUT_S) -> str:
    return f"[out:json][timeout:{timeout}];\n{statement}\nout count;"


def build_fetch_query(statement: str, timeout: int = TIMEOUT_S, limit: int = MAX_FETCH) -> str:
    return f"[out:json][timeout:{timeout}];\n{statement}\nout center {limit};"


def parse_count(payload: dict) -> int:
    for el in payload.get("elements", []):
        if el.get("type") == "count":
            try:
                return int(el.get("tags", {}).get("total", 0))
            except (TypeError, ValueError):
                break
    if "remark" in payload:
        raise overpass.OverpassError(f"Overpass reported: {payload['remark']}")
    raise overpass.OverpassError("Overpass returned no count")


def haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = p2 - p1
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * overpass.EARTH_RADIUS_KM * 1000 * math.asin(math.sqrt(a))


def _matched_field(tags: dict[str, str], pattern: re.Pattern[str]) -> tuple[str, str] | None:
    key_re = re.compile(NAME_KEY_REGEX)
    # Prefer the plain name keys, then language variants.
    keys = sorted((k for k in tags if key_re.match(k)), key=lambda k: (":" in k, k))
    for k in keys:
        if pattern.search(tags[k]):
            return k, tags[k]
    return None


def parse_elements(payload: dict, name: str, bbox: BBox) -> list[dict]:
    """Turn Overpass elements into results sorted by distance from the bbox centre."""
    pattern = re.compile(name_regex(name), re.IGNORECASE)
    clat, clon = bbox.center
    results = []
    for el in payload.get("elements", []):
        if el.get("type") not in ("node", "way", "relation"):
            continue
        if "lat" in el and "lon" in el:
            lat, lon = el["lat"], el["lon"]
        elif "center" in el:
            lat, lon = el["center"]["lat"], el["center"]["lon"]
        else:
            continue
        tags = el.get("tags", {})
        matched = _matched_field(tags, pattern)
        field_name, field_value = matched if matched else (None, None)
        results.append({
            "name": tags.get("name") or field_value,
            "matched_field": field_name,
            "matched_value": field_value,
            "osm_type": el["type"],
            "osm_id": el.get("id"),
            "osm_url": f"https://www.openstreetmap.org/{el['type']}/{el.get('id')}",
            "lat": lat,
            "lon": lon,
            "distance_to_bbox_center_m": round(haversine_m(clat, clon, lat, lon)),
            "tags": {k: tags[k] for k in MAIN_TAGS if k in tags},
        })
    results.sort(key=lambda r: r["distance_to_bbox_center_m"])
    return results


async def find_named_features(
    name: str, bbox: BBox, tags: dict[str, str] | None = None, max_results: int = MAX_RESULTS,
) -> dict:
    """Count matching features first, then fetch them if there are any."""
    overpass.validate_bbox(bbox.south, bbox.west, bbox.north, bbox.east, MAX_BBOX_KM2)
    name = validate_name(name)
    statement = build_statement(name, bbox, tags)

    total = parse_count(await overpass.run_raw(build_count_query(statement), TIMEOUT_S))
    result: dict = {
        "query_name": name,
        "total_found": total,
        "truncated": False,
        "results": [],
        "attribution": "© OpenStreetMap contributors (ODbL)",
    }
    if total == 0:
        result["note"] = (
            "No OpenStreetMap feature with a matching name in this box. Try a larger bbox, a shorter "
            "or more distinctive part of the name, fixing likely OCR misreadings (0/O, 1/l), or "
            "removing the tag filters. The place may also be missing from OpenStreetMap."
        )
        return result

    payload = await overpass.run_raw(build_fetch_query(statement), TIMEOUT_S)
    found = parse_elements(payload, name, bbox)
    result["results"] = found[:max_results]
    result["truncated"] = total > len(result["results"])
    if total > MAX_FETCH:
        result["note"] = (
            f"{total} matches; only {MAX_FETCH} were fetched, so the nearest ones shown may not be "
            "the nearest overall. Narrow the bbox, add tag filters or use a longer name."
        )
    return result
