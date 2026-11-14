"""The dating-clues table: parsing, the embedding cache key, and the semantic search.

No model and no network: the parser works on table text written inline, and the search is
driven by small deterministic vectors standing in for embeddings.
"""
import pytest

pytest.importorskip("numpy")

import numpy as np

from geotrace.core import dating_clues as dc
from geotrace.core import time_objects as to
from geotrace.tools import time_from_objects as tfo

HEADER = "| Clothing item or object | Appears or starts to spread | Peak | Clue type |\n|---|---|---|---|\n"


def table(*rows: str) -> str:
    return HEADER + "".join(f"{r}\n" for r in rows)


def one(row: str) -> dc.ClueRow:
    rows = dc.parse_table(table(row))
    assert len(rows) == 1
    return rows[0]


# --- the appears column -------------------------------------------------------------------

@pytest.mark.parametrize("text,expected", [
    ("2015", 2015),
    ("June 2015", 2015),
    ("December 2016", 2016),
    ("October 2001", 2001),
    ("Summer 2024", 2024),
    ("Late 2004", 2004),
    ("Early 1990s", 1990),
    ("Early 2000s", 2000),
    ("Mid-1980s", 1984),
    ("Mid-2000s", 2004),
    ("Late 1990s", 1997),
    ("Late 2000s", 2007),
    ("2021–2022", 2021),            # a range in this column: the first year is when it appears
    ("(already existed)", None),
])
def test_appears_year(text, expected):
    assert one(f"| Thing | {text} | 2020 | Weak |").appears_year == expected


# --- the peak column ----------------------------------------------------------------------

@pytest.mark.parametrize("text,expected", [
    ("2016–2017", (2016, 2017)),
    ("1970–1976", (1970, 1976)),
    ("2023–early 2024", (2023, 2024)),
    ("2015", (2015, 2015)),
    ("Summer 2023", (2023, 2023)),
    ("Spring and summer 2017", (2017, 2017)),
    ("2018 onwards", (2018, None)),
    ("2015 onwards", (2015, None)),
    ("1990s", (1990, 1999)),
    ("(ongoing)", (None, None)),
])
def test_peak_range(text, expected):
    row = one(f"| Thing | 2010 | {text} | Weak |")
    assert (row.peak_from, row.peak_to) == expected


def test_original_text_is_kept_alongside_the_numbers():
    row = one('| Thing | Mid-1980s | 1990s | Strong ("not before 1987") |')
    assert row.appears == "Mid-1980s" and row.appears_year == 1984
    assert row.peak == "1990s" and (row.peak_from, row.peak_to) == (1990, 1999)
    assert row.clue_type == 'Strong ("not before 1987")'


# --- strength and not_before_year ----------------------------------------------------------

@pytest.mark.parametrize("clue_type,strength", [
    ('Strong ("not before 1996")', "strong"),
    ('Medium-strong ("around 1997")', "medium-strong"),
    ('Medium ("2014 or later")', "medium"),
    ("Weak", "weak"),
])
def test_strength(clue_type, strength):
    assert one(f"| Thing | 2010 | 2012 | {clue_type} |").strength == strength


@pytest.mark.parametrize("clue_type,expected", [
    ('Strong ("not before 1979")', 1979),
    ('Strong ("not before late 2016")', 2016),
    ('Strong ("not before mid-2015")', 2015),
    ('Medium ("2014 or later")', 2014),
    ('Medium-strong ("2025 or later")', 2025),
    # "around" is an era, not a lower bound, so it yields no year.
    ('Medium ("around 1990–1991")', None),
    ('Medium-strong ("around 2004–2005")', None),
    # A strong clue that states no quoted bound keeps its text but gives no year.
    ("Strong (2020–2022 in Europe)", None),
    ("Weak", None),
])
def test_not_before_year(clue_type, expected):
    row = one(f"| Thing | 2010 | 2012 | {clue_type} |")
    assert row.not_before_year == expected
    assert row.clue_type == clue_type


def test_a_weak_row_never_gets_a_bound_even_if_it_quotes_a_year():
    assert one('| Thing | 2010 | 2012 | Weak ("not before 2010") |').not_before_year is None


# --- malformed rows -----------------------------------------------------------------------

@pytest.mark.parametrize("row,fragment", [
    ("| Thing | 2010 | 2012 |", "expected 4 columns"),
    ("| Thing | 2010 | 2012 | Enormous |", "unknown clue type"),
    ("| Thing | sometime | 2012 | Weak |", "appears column"),
    ("| Thing | 2010 | whenever | Weak |", "peak column"),
    ("| Thing | 2010 | 2012-2014 | Weak |", "ambiguous peak"),   # ASCII hyphen, not an en dash
    ("|  | 2010 | 2012 | Weak |", "item name is empty"),
])
def test_a_bad_row_raises_and_names_the_line(row, fragment):
    with pytest.raises(dc.ClueTableError) as err:
        dc.parse_table(table("| Good | 2000 | 2001 | Weak |", row))
    assert fragment in str(err.value)
    assert "line 4" in str(err.value)          # header, separator, the good row, then this one


def test_an_empty_table_raises():
    with pytest.raises(dc.ClueTableError):
        dc.parse_table(HEADER)


# --- the real table -----------------------------------------------------------------------

def test_the_real_table_parses():
    rows = dc.parse_table(dc.table_text())
    assert len(rows) > 150
    assert all(r.item for r in rows)
    assert all(r.strength in ("strong", "medium-strong", "medium", "weak") for r in rows)
    # Every bound the table states is inside the range it covers.
    assert all(1970 <= r.not_before_year <= 2026
               for r in rows if r.not_before_year is not None)
    # Only strong and medium clues carry a bound.
    assert all(r.not_before_year is None for r in rows if r.strength == "weak")
    assert any(r.not_before_year is not None for r in rows)


# --- the cache key ------------------------------------------------------------------------

def test_cache_key_changes_with_the_table_text():
    a = dc.cache_key("| A | 2010 | 2011 | Weak |")
    b = dc.cache_key("| B | 2010 | 2011 | Weak |")
    assert a != b


def test_cache_key_changes_with_the_model():
    text = "| A | 2010 | 2011 | Weak |"
    base = dc.cache_key(text)
    assert dc.cache_key(text, model_name="ViT-B-32") != base
    assert dc.cache_key(text, pretrained="laion2b_s34b_b79k") != base


def test_cache_key_is_stable():
    assert dc.cache_key("| A | 2010 | 2011 | Weak |") == dc.cache_key("| A | 2010 | 2011 | Weak |")


# --- centring and the background statistics ------------------------------------------------

def test_cache_key_changes_with_the_background_queries():
    text = "| A | 2010 | 2011 | Weak |"
    assert dc.cache_key(text, background=("a dog",)) != dc.cache_key(text, background=("a cat",))


@pytest.fixture
def stub_embeddings(monkeypatch):
    """Replace the model with a lookup table from text to a chosen vector."""
    def use(vectors: dict[str, list[float]]):
        def _embed_rows(texts):
            return np.asarray([vectors[t] for t in texts], dtype=np.float32)
        monkeypatch.setattr(dc, "_embed_rows", _embed_rows)

    return use


def test_centring_removes_the_row_mean_and_renormalises(stub_embeddings):
    rows = dc.parse_table(table(
        "| A | 2010 | 2011 | Weak |",
        "| B | 2010 | 2011 | Weak |",
    ))
    # Both rows share a large component on axis 0: that shared direction is the hub.
    stub_embeddings({"A": [1.0, 0.2, 0.0], "B": [1.0, 0.0, 0.2],
                     **{b: [1.0, 0.0, 0.0] for b in dc.BACKGROUND_QUERIES}})
    index = dc.build_index(rows)

    assert index["raw"].shape == (2, 3)
    # The mean row is the shared direction, and centring takes it out.
    assert index["mean_row"] == pytest.approx([1.0, 0.1, 0.1])
    assert np.linalg.norm(index["centred"], axis=1) == pytest.approx([1.0, 1.0])
    assert index["centred"].mean(axis=0) == pytest.approx([0.0, 0.0, 0.0], abs=1e-6)


def test_background_statistics_are_measured_per_row(stub_embeddings):
    rows = dc.parse_table(table(
        "| Hub | 2010 | 2011 | Weak |",
        "| Specific | 2010 | 2011 | Weak |",
    ))
    # The background queries all sit on the Hub row's side of the centred space.
    stub_embeddings({"Hub": [1.0, 0.0], "Specific": [0.0, 1.0],
                     **{b: [1.0, 0.0] for b in dc.BACKGROUND_QUERIES}})
    index = dc.build_index(rows)

    hub, specific = index["background_mean"]
    assert hub > 0 > specific               # the hub row looks like everything
    assert index["background_std"].min() > 0  # never zero, so z stays finite


def test_a_hub_row_needs_a_higher_cosine_to_reach_the_same_z(stub_embeddings):
    """The whole point of the z-score: being everyone's neighbour stops being an advantage."""
    rows = dc.parse_table(table(
        "| Hub | 2010 | 2011 | Weak |",
        "| Specific | 2010 | 2011 | Weak |",
    ))
    stub_embeddings({"Hub": [1.0, 0.0], "Specific": [0.0, 1.0],
                     **{b: [0.98, 0.2] for b in dc.BACKGROUND_QUERIES}})
    index = dc.build_index(rows)
    assert index["background_mean"][0] > index["background_mean"][1]


def test_queries_are_centred_and_renormalised(monkeypatch):
    """A query goes through the same centring as the rows before it is scored."""
    rows = dc.parse_table(table("| A | 2010 | 2011 | Weak |"))
    index = {
        "raw": np.asarray([[1.0, 0.0]], dtype=np.float32),
        "centred": np.asarray([[1.0, 0.0]], dtype=np.float32),
        "mean_row": np.asarray([0.0, 0.6], dtype=np.float32),
        "background_mean": np.zeros(1, dtype=np.float32),
        "background_std": np.ones(1, dtype=np.float32),
    }
    monkeypatch.setattr(dc, "table_text", lambda: table("| A | 2010 | 2011 | Weak |"))
    monkeypatch.setattr(dc, "row_embeddings", lambda r, t: index)
    monkeypatch.setattr(dc, "_load", lambda: (None, None, None))
    monkeypatch.setattr(dc, "_MEMORY_CACHE", {})
    monkeypatch.setattr(dc, "Z_MATCH_THRESHOLD", 0.0)
    # The query is (0.6, 0.6); the mean row leaves (0.6, 0.0), which renormalises to (1, 0).
    monkeypatch.setattr(dc, "_embed_text",
                        lambda m, t, texts: np.asarray([[0.6, 0.6]], dtype=np.float32))

    match = dc.match_objects(["A"], top_k=1)["A"][0]
    assert match.z == pytest.approx(1.0)          # centred and renormalised, not 0.6
    assert match.similarity == pytest.approx(0.6)  # the plain cosine is reported as-is
    assert len(rows) == 1


# --- the lexical gate ----------------------------------------------------------------------

@pytest.mark.parametrize("query,item,shared", [
    ("flip phone", "Motorola RAZR flip phone", True),
    ("airpods", "AirPods wireless earbuds", True),          # case is ignored
    ("velour tracksuit", "Velour tracksuits (Juicy Couture style)", True),   # plural
    ("football shirt", "Retro football shirts as fashion", True),
    ("labubu", "Labubu plush toys on bags", True),
    ("crocs", "Classic Crocs", True),
    ("a crocodile", "Classic Crocs", False),                # the z-score's worst false match
    ("cathedral", "Quiet luxury", False),
    ("wireless earbuds", "Bluetooth earpieces", False),
    # A shared filler word must never count as agreement.
    ("a photo worn with the style", "Overalls worn with one strap down", False),
])
def test_shares_words(query, item, shared):
    assert dc._shares_words(query, item) is shared


def test_words_drops_filler_and_short_tokens():
    assert dc._words("Bell-bottom jeans and trousers") == {"bell", "bottom", "jean", "trouser"}


# --- the search, on fake embeddings --------------------------------------------------------

ROWS = table(
    '| Motorola RAZR flip phone | Late 2004 | 2005–2007 | Strong ("not before late 2004") |',
    '| AirPods wireless earbuds | December 2016 | 2018 onwards | Strong ("not before late 2016") |',
    "| Bell-bottom jeans | Late 1960s | 1970–1976 | Weak |",
)
AXES = np.eye(3, dtype=np.float32)


def sims(*values: float) -> np.ndarray:
    """A unit query whose cosine to the three rows is exactly `values`."""
    return np.asarray(values, dtype=np.float32)


@pytest.fixture
def fake_table(monkeypatch):
    """The three rows above on three unit axes, with neutral background statistics.

    With background mean 0 and standard deviation 1, z equals the cosine, so a query vector
    is its own list of scores. _unitise is stubbed out so those scores survive unchanged;
    the real normalisation is covered by the centring tests above and by
    test_queries_are_centred_and_renormalised. The real Z_MATCH_THRESHOLD is a calibration
    constant rather than logic, so it is scaled down here to keep the vectors readable.
    """
    index = {
        "raw": AXES,
        "centred": AXES,
        "mean_row": np.zeros(3, dtype=np.float32),
        "background_mean": np.zeros(3, dtype=np.float32),
        "background_std": np.ones(3, dtype=np.float32),
    }
    monkeypatch.setattr(dc, "table_text", lambda: ROWS)
    monkeypatch.setattr(dc, "row_embeddings", lambda rows, text: index)
    monkeypatch.setattr(dc, "_load", lambda: (None, None, None))
    monkeypatch.setattr(dc, "_MEMORY_CACHE", {})
    monkeypatch.setattr(dc, "_unitise", lambda vectors: vectors)
    monkeypatch.setattr(dc, "Z_MATCH_THRESHOLD", 0.5)
    monkeypatch.setattr(dc, "IMAGE_SUGGESTION_THRESHOLD", 0.18)

    def queries(by_name: dict[str, np.ndarray]):
        def _embed_text(model, tokenizer, texts):
            return np.stack([by_name[t] for t in texts])
        monkeypatch.setattr(dc, "_embed_text", _embed_text)

    return queries


def q(name: str) -> str:
    return dc.ROW_TEMPLATE.format(item=name)


def test_a_shared_word_makes_a_match_confident(fake_table):
    fake_table({q("flip phone"): sims(0.95, 0.10, 0.10)})
    matches = dc.match_objects(["flip phone"])["flip phone"]
    assert [m.row.item for m in matches] == ["Motorola RAZR flip phone"]
    assert matches[0].confident is True
    assert matches[0].z == pytest.approx(0.95, abs=0.01)


def test_a_semantic_match_with_no_shared_word_is_only_a_candidate(fake_table):
    # Points straight at the RAZR row, but "handset" shares no word with it.
    fake_table({q("handset"): sims(0.95, 0.10, 0.10)})
    matches = dc.match_objects(["handset"])["handset"]
    assert [m.row.item for m in matches] == ["Motorola RAZR flip phone"]
    assert matches[0].confident is False


def test_a_row_below_the_threshold_is_not_returned(fake_table):
    fake_table({q("cathedral"): sims(0.30, 0.20, 0.10)})
    assert dc.match_objects(["cathedral"])["cathedral"] == []


def test_rows_are_ranked_by_z_not_by_raw_cosine(fake_table, monkeypatch):
    """A hub row with a higher cosine must lose to a row that stands out from its baseline."""
    index = {
        "raw": AXES,
        "centred": AXES,
        "mean_row": np.zeros(3, dtype=np.float32),
        # The RAZR row is a hub: it sits 0.80 from anything by default.
        "background_mean": np.asarray([0.80, 0.0, 0.0], dtype=np.float32),
        "background_std": np.ones(3, dtype=np.float32),
    }
    monkeypatch.setattr(dc, "row_embeddings", lambda rows, text: index)
    fake_table({q("earbuds"): sims(0.90, 0.70, 0.10)})

    matches = dc.match_objects(["earbuds"])["earbuds"]
    # Raw cosine would put the RAZR row first (0.90 > 0.70); z puts the AirPods row first.
    assert matches[0].row.item == "AirPods wireless earbuds"
    assert matches[0].similarity == pytest.approx(0.70, abs=0.01)
    assert matches[0].z == pytest.approx(0.70, abs=0.01)


def test_confidence_is_not_limited_to_the_top_row(fake_table):
    """The lexically matching row can sit below a closer but unrelated neighbour."""
    fake_table({q("wireless earbuds"): sims(0.95, 0.80, 0.10)})
    matches = dc.match_objects(["wireless earbuds"])["wireless earbuds"]
    assert [m.row.item for m in matches] == [
        "Motorola RAZR flip phone", "AirPods wireless earbuds"]
    assert [m.confident for m in matches] == [False, True]


def test_top_k_limits_the_rows_returned(fake_table):
    fake_table({q("thing"): sims(0.95, 0.90, 0.85)})
    assert len(dc.match_objects(["thing"], top_k=3)["thing"]) == 3
    assert len(dc.match_objects(["thing"], top_k=1)["thing"]) == 1


def test_every_name_gets_an_entry(fake_table):
    fake_table({q("flip phone"): sims(0.95, 0.10, 0.10),
                q("cathedral"): sims(0.30, 0.20, 0.10)})
    matches = dc.match_objects(["flip phone", "cathedral"])
    assert set(matches) == {"flip phone", "cathedral"}
    assert matches["cathedral"] == []
    assert len(matches["flip phone"]) == 1


def test_image_suggestions_use_the_raw_cosine_and_report_no_z(fake_table):
    fake_table({})
    hits = dc.suggest_from_image(sims(0.05, 0.50, 0.05))
    assert [h.row.item for h in hits] == ["AirPods wireless earbuds"]
    assert hits[0].similarity == pytest.approx(0.50, abs=0.01)
    assert hits[0].z is None                   # no text background applies to an image


def test_a_weak_image_resemblance_suggests_nothing(fake_table):
    fake_table({})
    assert dc.suggest_from_image(sims(0.10, 0.10, 0.10)) == []


def test_image_suggestions_are_never_marked_confident(fake_table):
    fake_table({})
    hits = dc.suggest_from_image(sims(0.95, 0.90, 0.85))
    assert len(hits) == 3
    assert all(not h.confident for h in hits)


# --- the assembled signal -----------------------------------------------------------------

def test_lookup_bounds_the_date_from_a_confident_match(fake_table):
    fake_table({q("flip phone"): sims(0.95, 0.10, 0.10)})
    out = dc.lookup(["flip phone"], sims(0.10, 0.10, 0.10))
    assert out["table_not_before_year"] == 2004
    match = out["matches"]["flip phone"][0]
    assert match["item"] == "Motorola RAZR flip phone"
    assert match["clue_type"] == 'Strong ("not before late 2004")'
    assert match["appears"] == "Late 2004"
    assert match["peak"] == "2005–2007"
    assert match["not_before_year"] == 2004


def test_lookup_takes_the_latest_of_several_confident_matches(fake_table):
    fake_table({q("flip phone"): sims(0.95, 0.10, 0.10),
                q("airpods"): sims(0.10, 0.95, 0.10)})
    out = dc.lookup(["flip phone", "airpods"], sims(0.10, 0.10, 0.10))
    assert out["table_not_before_year"] == 2016


def test_lookup_ignores_image_suggestions(fake_table):
    # The image looks exactly like the AirPods row, but nothing matched by name.
    fake_table({q("cathedral"): sims(0.30, 0.20, 0.10)})
    out = dc.lookup(["cathedral"], sims(0.05, 0.95, 0.05))
    assert [s["item"] for s in out["image_suggestions"]] == ["AirPods wireless earbuds"]
    assert out["matches"]["cathedral"] == []
    assert out["table_not_before_year"] is None          # suggestions are unverified


def test_lookup_ignores_a_candidate_with_no_shared_word(fake_table):
    """A strong row reached only semantically must not date the photo."""
    fake_table({q("handset"): sims(0.95, 0.10, 0.10)})
    out = dc.lookup(["handset"], sims(0.10, 0.10, 0.10))
    assert out["matches"]["handset"][0]["not_before_year"] == 2004
    assert out["matches"]["handset"][0]["confident"] is False
    assert out["table_not_before_year"] is None


def test_lookup_ignores_a_confident_match_with_no_bound(fake_table):
    # The bell-bottoms row is a weak clue, so it carries no "not before" year.
    fake_table({q("bell-bottom jeans"): sims(0.10, 0.10, 0.95)})
    out = dc.lookup(["bell-bottom jeans"], sims(0.10, 0.10, 0.10))
    assert out["matches"]["bell-bottom jeans"][0]["confident"] is True
    assert out["matches"]["bell-bottom jeans"][0]["not_before_year"] is None
    assert out["table_not_before_year"] is None


def test_match_dicts_carry_only_the_documented_fields(fake_table):
    fake_table({q("flip phone"): sims(0.95, 0.10, 0.10)})
    out = dc.lookup(["flip phone"], sims(0.95, 0.10, 0.10))
    expected = {"item", "appears", "peak", "clue_type",
                "not_before_year", "similarity", "z", "confident"}
    assert set(out["matches"]["flip phone"][0]) == expected
    assert set(out["image_suggestions"][0]) == expected


def test_the_calibrated_constants_are_the_ones_that_were_measured():
    """A guard on the calibration: changing these needs a re-run, not a guess."""
    assert dc.Z_MATCH_THRESHOLD == 5.0
    assert dc.IMAGE_SUGGESTION_THRESHOLD == 0.18
    assert len(dc.BACKGROUND_QUERIES) == 33


# --- how core and the tool report the signal ------------------------------------------------

CLUES = {
    "matches": {"flip phone": [{
        "item": "Motorola RAZR flip phone", "appears": "Late 2004", "peak": "2005–2007",
        "clue_type": 'Strong ("not before late 2004")', "not_before_year": 2004,
        "similarity": 0.91, "confident": True}]},
    "image_suggestions": [],
    "table_not_before_year": 2004,
}


class FakeTensor:
    """Just enough of a torch tensor for core.time_objects: matmul, .T and detach/cpu/numpy."""

    def __init__(self, values):
        self.values = np.asarray(values, dtype=np.float32)

    @property
    def T(self):
        return FakeTensor(self.values.T)

    def __matmul__(self, other):
        return FakeTensor(self.values @ other.values)

    def detach(self):
        return self

    def cpu(self):
        return self

    def numpy(self):
        return self.values


@pytest.fixture
def fake_core(monkeypatch):
    """core.time_objects with the EVA02 model replaced by a stub, so nothing is downloaded."""
    image = FakeTensor([[1.0]])
    monkeypatch.setattr(to, "_load", lambda: ("model", "preprocess", "tokenizer"))
    monkeypatch.setattr(to, "_embed_image", lambda model, preprocess, data: image)

    def fake_text(model, tokenizer, texts):
        # One column per candidate year; only the 1990 prompt matches the image, so the build
        # year comes out as 1990 and built_from_year as 1990 minus the tolerance.
        return FakeTensor([[1.0 if "1990" in t else 0.0] for t in texts])

    monkeypatch.setattr(to, "_embed_text", fake_text)
    return image


def test_core_reuses_one_image_embedding_for_both_signals(fake_core, monkeypatch):
    """The image is encoded once, and that same embedding is what reaches the table lookup."""
    encoded = []
    monkeypatch.setattr(
        to, "_embed_image",
        lambda model, preprocess, data: encoded.append(data) or fake_core)
    seen = {}

    def lookup(names, embedding):
        seen["names"] = names
        seen["embedding"] = embedding
        return CLUES

    monkeypatch.setattr(dc, "lookup", lookup)
    out = to.time_from_objects(b"jpeg", ["flip phone"])

    assert encoded == [b"jpeg"]                    # encoded exactly once
    assert seen["embedding"] is fake_core          # and reused, not encoded a second time
    assert seen["names"] == ["flip phone"]
    assert out["dating_clues"] == CLUES


def test_core_keeps_the_object_intervals_when_the_table_fails(fake_core, monkeypatch):
    def boom(names, embedding):
        raise FileNotFoundError("the dating-clues table was not found")

    monkeypatch.setattr(dc, "lookup", boom)
    out = to.time_from_objects(b"jpeg", ["flip phone"])

    assert "dating_clues" not in out
    assert "FileNotFoundError" in out["dating_clues_error"]
    assert "not found" in out["dating_clues_error"]
    # The other signal from this function is untouched: the build year the stub steers to,
    # less the paper's tolerance at that year.
    tol = float(np.interp(1990, to.TOL_YEARS, to.TOL_VALUES))
    assert out["image_taken_not_before_year"] == int(np.floor(1990 - tol))
    assert out["objects"][0]["object"] == "flip phone"


class FakeContext:
    def __init__(self):
        self.messages = []

    async def info(self, message):
        self.messages.append(message)


INTERVALS = {
    "current_year": 2026,
    "image_taken_not_before_year": 2004,
    "objects": [{"object": "flip phone", "built_from_year": 2004,
                 "built_until_year": 2026, "tolerance_years": 5.0}],
    "dating_clues": CLUES,
}
DATING = {"estimated_year": 2005, "expected_year": 2005.2, "interval_80pct": [2000, 2010],
          "confidence": 0.3, "model_year_range": [1950, 2010]}


@pytest.fixture
def tool(monkeypatch):
    async def fake_load(image):
        return b"jpeg"

    monkeypatch.setattr(tfo, "load_image", fake_load)
    monkeypatch.setattr(tfo, "row_embeddings_cached", lambda: True)
    monkeypatch.setattr(tfo, "estimate_intervals", lambda data, objects: dict(INTERVALS))
    monkeypatch.setattr(tfo, "estimate_capture_year", lambda data: dict(DATING))


async def run(ctx=None):
    return await tfo.time_from_objects(
        image="photo.jpg", objects=["flip phone"],
        ctx=ctx or FakeContext())  # type: ignore[arg-type]


async def test_tool_returns_the_table_signal(tool):
    out = await run()
    clues = out["dating_clues"]
    assert clues["table_not_before_year"] == 2004
    assert clues["matches"]["flip phone"][0]["item"] == "Motorola RAZR flip phone"
    assert clues["image_suggestions"] == []
    assert "dating_clues_error" not in out


async def test_tool_keeps_the_other_two_signals(tool):
    out = await run()
    # The table is a separate signal and must not be folded into the object bound.
    assert out["image_taken_not_before_year"] == 2004
    assert out["current_year"] == 2026
    assert out["objects"][0]["built_from_year"] == 2004
    assert out["capture_year_estimate"]["estimated_year"] == 2005


async def test_tool_reports_a_failed_lookup_without_losing_the_rest(tool, monkeypatch):
    broken = {k: v for k, v in INTERVALS.items() if k != "dating_clues"}
    broken["dating_clues_error"] = "ClueTableError: line 7: unknown clue type 'Enormous'"
    monkeypatch.setattr(tfo, "estimate_intervals", lambda data, objects: dict(broken))

    ctx = FakeContext()
    out = await run(ctx)
    assert "dating_clues" not in out
    assert "unknown clue type" in out["dating_clues_error"]
    assert out["image_taken_not_before_year"] == 2004
    assert out["capture_year_estimate"]["estimated_year"] == 2005
    assert any("could not be searched" in m for m in ctx.messages)


async def test_tool_warns_before_embedding_the_table_for_the_first_time(tool, monkeypatch):
    monkeypatch.setattr(tfo, "row_embeddings_cached", lambda: False)
    ctx = FakeContext()
    await run(ctx)
    assert any("first time" in m for m in ctx.messages)


async def test_tool_does_not_warn_when_the_embeddings_are_cached(tool):
    ctx = FakeContext()
    await run(ctx)
    assert not any("first time" in m for m in ctx.messages)


async def test_tool_survives_a_missing_table_when_checking_the_cache(tool, monkeypatch):
    def boom():
        raise FileNotFoundError("no table")

    monkeypatch.setattr(tfo, "row_embeddings_cached", boom)
    ctx = FakeContext()
    out = await run(ctx)
    assert out["dating_clues"]["table_not_before_year"] == 2004
    assert any("Could not check" in m for m in ctx.messages)
