import asyncio
import re

import pytest

from geotrace.core import named_features as nf
from geotrace.core import overpass

BOX = nf.BBox(south=41.89, west=12.47, north=41.91, east=12.49)


def ql_unquote(literal: str) -> str:
    """Inverse of ql_string, to check the regex Overpass would receive."""
    assert literal[0] == literal[-1] == '"'
    out, i, body = [], 0, literal[1:-1]
    while i < len(body):
        if body[i] == "\\":
            out.append(body[i + 1])
            i += 2
        else:
            assert body[i] != '"', "unescaped quote would end the QL string"
            out.append(body[i])
            i += 1
    return "".join(out)


def value_regex(statement: str) -> str:
    m = re.search(r'\[~"(?:[^"\\]|\\.)*"~("(?:[^"\\]|\\.)*"),i\]', statement)
    assert m, statement
    return ql_unquote(m.group(1))


# --- Escaping ----------------------------------------------------------------------------

SPECIALS = ['.', '*', '(', ')', '"', '\\', '[', ']', '{', '}', '+', '?', '|', '^', '$']


@pytest.mark.parametrize("ch", SPECIALS)
def test_special_characters_are_matched_literally(ch):
    text = f"Bar {ch}X{ch} 24"
    regex = value_regex(nf.build_statement(text, BOX))
    pattern = re.compile(regex, re.IGNORECASE)
    assert pattern.search(f"Il {text} Roma")
    # The special character must not act as a metacharacter.
    assert not pattern.search("Bar QXQ 24") or ch == "Q"


def test_escape_ere():
    assert nf.escape_ere('a.b*c(d)"e\\f') == 'a\\.b\\*c\\(d\\)"e\\\\f'


def test_ql_string_escapes_quotes_and_backslashes():
    assert nf.ql_string('say "hi" \\o/') == '"say \\"hi\\" \\\\o/"'


def test_statement_cannot_break_out_of_string():
    stmt = nf.build_statement('x"];out;node(1);("', BOX)
    # The whole statement keeps one nwr selection, and the payload stays inside the string.
    assert stmt.startswith("nwr[~") and stmt.endswith(f"({BOX.ql()});")
    assert stmt.count(";") == 4  # the closing one plus the three inside the quoted regex
    assert re.compile(value_regex(stmt), re.IGNORECASE).search('x"];out;node(1);("')


def test_control_characters_and_whitespace_collapsed():
    assert nf.normalize_name("  Via\t\n  Roma\x00 ") == "Via Roma"
    regex = value_regex(nf.build_statement("Via \n  Roma", BOX))
    assert re.search(regex, "via roma", re.IGNORECASE)
    assert "\n" not in regex


# --- Approximate matching ----------------------------------------------------------------

def test_accent_insensitive_both_ways():
    with_accent = re.compile(nf.name_regex("Café de la Gare"), re.IGNORECASE)
    without = re.compile(nf.name_regex("Cafe de la Gare"), re.IGNORECASE)
    for p in (with_accent, without):
        assert p.search("CAFÉ DE LA GARE")
        assert p.search("cafe de la gare")
        assert p.search("Le Café de la Gare")
        assert not p.search("Cafe du Port")


def test_accent_free_variant_used():
    regex = nf.name_regex("Plaza Mayor Señorío")
    assert "ñ" in regex and "Ñ" in regex
    assert re.search(regex, "PLAZA MAYOR SENORIO", re.IGNORECASE)


def test_non_latin_original_kept():
    # NFKD turns Cyrillic й into и + breve; the original must still match.
    p = re.compile(nf.name_regex("Музей"), re.IGNORECASE)
    assert p.search("Музей истории")


def test_name_length_limits():
    with pytest.raises(nf.NamedFeatureError):
        nf.build_statement(" a ", BOX)
    with pytest.raises(nf.NamedFeatureError):
        nf.build_statement("x" * 101, BOX)


# --- Query structure ---------------------------------------------------------------------

def test_name_keys_regex():
    stmt = nf.build_statement("Roma", BOX)
    m = re.search(r'\[~("(?:[^"\\]|\\.)*")~', stmt)
    assert m
    keys = re.compile(ql_unquote(m.group(1)))
    for k in ["name", "name:en", "name:zh-Hant", "alt_name", "old_name", "official_name", "alt_name:it"]:
        assert keys.match(k), k
    for k in ["brand", "addr:street", "short_name", "name:etymology:wikidata", "xname"]:
        assert not keys.match(k), k
    assert ",i]" in stmt  # case-insensitive


def test_bbox_order_south_west_north_east():
    box = nf.BBox(south=1.5, west=2.5, north=3.5, east=4.5)
    assert nf.build_statement("Roma", box).endswith("(1.5,2.5,3.5,4.5);")


def test_tag_filters():
    stmt = nf.build_statement("Roma", BOX, {"amenity": "pharmacy", "wheelchair": "*", 'we"ird': 'v\\'})
    assert stmt.startswith('nwr["amenity"="pharmacy"]["wheelchair"]["we\\"ird"="v\\\\"][~')


def test_too_many_tag_filters():
    with pytest.raises(nf.NamedFeatureError):
        nf.build_statement("Roma", BOX, {str(i): "x" for i in range(6)})


def test_count_and_fetch_queries():
    stmt = nf.build_statement("Roma", BOX)
    assert nf.build_count_query(stmt) == f"[out:json][timeout:25];\n{stmt}\nout count;"
    assert nf.build_fetch_query(stmt).endswith(f"\nout center {nf.MAX_FETCH};")


# --- Parsing and the full flow (Overpass mocked) -----------------------------------------

def count_payload(total):
    return {"elements": [{"type": "count", "id": 0, "tags": {"nodes": str(total), "total": str(total)}}]}


ELEMENTS = {"elements": [
    {"type": "way", "id": 2, "center": {"lat": 41.905, "lon": 12.48},
     "tags": {"name": "Farmacia Centrale", "amenity": "pharmacy", "opening_hours": "24/7"}},
    {"type": "node", "id": 1, "lat": 41.9, "lon": 12.48,
     "tags": {"name": "Bar Sport", "name:it": "Farmacia Vecchia", "shop": "chemist"}},
    {"type": "relation", "id": 3, "tags": {"name": "no geometry"}},
]}


def test_parse_elements_sorted_by_distance():
    out = nf.parse_elements(ELEMENTS, "farmacia", BOX)
    assert [r["osm_id"] for r in out] == [1, 2]
    assert out[0]["distance_to_bbox_center_m"] == 0
    assert out[0]["matched_field"] == "name:it"
    assert out[0]["matched_value"] == "Farmacia Vecchia"
    assert out[0]["name"] == "Bar Sport"
    assert out[1]["lat"] == 41.905 and out[1]["osm_url"] == "https://www.openstreetmap.org/way/2"
    assert out[1]["tags"] == {"amenity": "pharmacy"}
    assert 500 < out[1]["distance_to_bbox_center_m"] < 600


class FakeOverpass:
    def __init__(self, *payloads):
        self.payloads = list(payloads)
        self.queries = []

    async def __call__(self, query, timeout=25):
        self.queries.append(query)
        return self.payloads.pop(0)


def test_find_counts_first_then_fetches(monkeypatch):
    fake = FakeOverpass(count_payload(2), ELEMENTS)
    monkeypatch.setattr(overpass, "run_raw", fake)
    result = asyncio.run(nf.find_named_features("Farmacia", BOX))
    assert fake.queries[0].endswith("out count;")
    assert "out center" in fake.queries[1]
    assert result["total_found"] == 2
    assert not result["truncated"]
    assert len(result["results"]) == 2
    assert "OpenStreetMap" in result["attribution"]


def test_find_nothing_found_returns_note(monkeypatch):
    fake = FakeOverpass(count_payload(0))
    monkeypatch.setattr(overpass, "run_raw", fake)
    result = asyncio.run(nf.find_named_features("Nowhere", BOX))
    assert len(fake.queries) == 1
    assert result["results"] == [] and result["total_found"] == 0
    assert "larger bbox" in result["note"]


def test_find_truncates_to_25(monkeypatch):
    many = {"elements": [
        {"type": "node", "id": i, "lat": 41.9 + i / 10000, "lon": 12.48, "tags": {"name": f"Roma {i}"}}
        for i in range(40)
    ]}
    monkeypatch.setattr(overpass, "run_raw", FakeOverpass(count_payload(40), many))
    result = asyncio.run(nf.find_named_features("roma", BOX))
    assert len(result["results"]) == 25
    assert result["truncated"]
    assert [r["osm_id"] for r in result["results"]] == list(range(25))


def test_find_rejects_large_bbox_before_querying(monkeypatch):
    fake = FakeOverpass()
    monkeypatch.setattr(overpass, "run_raw", fake)
    with pytest.raises(overpass.OverpassError, match="km²"):
        asyncio.run(nf.find_named_features("Roma", nf.BBox(41.0, 12.0, 42.0, 13.0)))
    assert fake.queries == []


def test_count_remark_raises(monkeypatch):
    monkeypatch.setattr(overpass, "run_raw", FakeOverpass({"elements": [], "remark": "runtime error: timeout"}))
    with pytest.raises(overpass.OverpassError, match="timeout"):
        asyncio.run(nf.find_named_features("Roma", BOX))
