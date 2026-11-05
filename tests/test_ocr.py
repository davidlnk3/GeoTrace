import io

import pytest
from PIL import Image

from geotrace.core import ocr
from geotrace.core.ocr import OCRError, TextBlock


def quad(x1, y1, x2, y2):
    return [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]


def png(width=1000, height=500) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (width, height), "white").save(buf, format="PNG")
    return buf.getvalue()


class FakeBackend:
    def __init__(self, languages, detections=()):
        self.languages = languages
        self.detections = list(detections)
        self.seen_size = None
        self.rotated = None

    def read(self, image, rotated=False):
        self.seen_size = image.size
        self.rotated = rotated
        return self.detections


@pytest.fixture(autouse=True)
def clear_readers(monkeypatch):
    monkeypatch.setattr(ocr, "_readers", {})


# --- Languages ---------------------------------------------------------------------------

@pytest.mark.parametrize("value", [None, [], ["", "  "]])
def test_default_languages(value):
    langs = ocr.normalize_languages(value)
    assert langs == ocr.DEFAULT_LANGUAGES
    assert {"en", "es", "fr", "de", "it", "pt", "nl"} <= set(langs)
    # The default set must be loadable together: all Latin-script.
    assert set(langs) <= set(ocr.LATIN)


def test_default_languages_is_a_copy():
    ocr.normalize_languages(None).append("xx")
    assert "xx" not in ocr.DEFAULT_LANGUAGES


@pytest.mark.parametrize("value,expected", [
    (["EN", " es ", "en"], ["en", "es"]),
    (["ru", "uk", "en"], ["ru", "uk", "en"]),
    (["ar", "fa", "en"], ["ar", "fa", "en"]),
    (["ja", "en"], ["ja", "en"]),
    (["ja"], ["ja"]),
    (["ch_sim", "en"], ["ch_sim", "en"]),
    (["hi", "mr", "en"], ["hi", "mr", "en"]),
    (["pl", "cs", "tr", "vi"], ["pl", "cs", "tr", "vi"]),
])
def test_explicit_languages(value, expected):
    assert ocr.normalize_languages(value) == expected


@pytest.mark.parametrize("value,script,rejected", [
    (["ja", "ko"], "Japanese", "ko"),
    (["ch_sim", "ja"], "Simplified Chinese", "ja"),
    (["ch_sim", "ch_tra"], "Traditional Chinese", "ch_sim"),
    (["fr", "ru"], "Cyrillic", "fr"),
    (["ar", "ru"], "Arabic", "ru"),
    (["th", "es"], "Thai", "es"),
    (["hi", "de", "en"], "Devanagari", "de"),
])
def test_invalid_combinations(value, script, rejected):
    with pytest.raises(OCRError) as exc:
        ocr.normalize_languages(value)
    msg = str(exc.value)
    assert script in msg
    assert rejected in msg
    assert "separate calls" in msg


def test_unknown_language():
    with pytest.raises(OCRError, match="zh"):
        ocr.normalize_languages(["zh", "en"])


# --- Post-processing ---------------------------------------------------------------------

def test_confidence_filter():
    dets = [(quad(0, 0, 100, 20), "KEEP", 0.9), (quad(0, 100, 100, 120), "DROP", 0.2)]
    blocks, total = ocr.process_detections(dets, (1000, 1000), min_confidence=0.3)
    assert [b.text for b in blocks] == ["KEEP"]
    assert total == 1


def test_empty_and_whitespace_text_dropped():
    dets = [(quad(0, 0, 100, 20), "  ", 0.9), (quad(0, 100, 100, 120), "A  B", 0.9)]
    blocks, _ = ocr.process_detections(dets, (1000, 1000))
    assert [b.text for b in blocks] == ["A B"]


def test_sorted_by_score_big_text_first():
    dets = [
        (quad(10, 900, 60, 910), "small print", 0.99),
        (quad(100, 100, 700, 250), "FARMACIA", 0.8),
        (quad(100, 400, 300, 450), "medium", 0.9),
    ]
    blocks, _ = ocr.process_detections(dets, (1000, 1000))
    assert [b.text for b in blocks] == ["FARMACIA", "medium", "small print"]
    assert blocks[0].relative_size == pytest.approx(600 * 150 / 1e6)
    assert blocks[0].bbox == (100, 100, 700, 250)


def test_confidence_breaks_ties_in_size():
    dets = [(quad(0, 0, 100, 50), "low", 0.4), (quad(0, 200, 100, 250), "high", 0.95)]
    blocks, _ = ocr.process_detections(dets, (1000, 1000))
    assert [b.text for b in blocks] == ["high", "low"]


def test_truncation_at_50():
    # One block per row, far apart so nothing merges.
    dets = [(quad(0, i * 40, 50, i * 40 + 20), f"t{i}", 0.5 + i / 1000) for i in range(70)]
    blocks, total = ocr.process_detections(dets, (1000, 3000))
    assert total == 70
    assert len(blocks) == ocr.MAX_BLOCKS == 50


def test_bbox_from_rotated_quad_is_clamped():
    points = [[-5, 10], [120, -3], [130, 40], [5, 52.4]]
    blocks, _ = ocr.process_detections([(points, "x y", 0.9)], (100, 50))
    assert blocks[0].bbox == (0, 0, 100, 50)


# --- Line grouping -----------------------------------------------------------------------

def tb(text, x1, y1, x2, y2, conf=0.9):
    return TextBlock(text=text, confidence=conf, bbox=(x1, y1, x2, y2), relative_size=0.0)


def test_groups_words_on_same_line():
    dets = [(quad(240, 100, 400, 140), "ROMA", 0.8), (quad(100, 100, 220, 140), "VIA", 0.95)]
    blocks, total = ocr.process_detections(dets, (1000, 1000))
    assert total == 1
    assert blocks[0].text == "VIA ROMA"
    assert blocks[0].bbox == (100, 100, 400, 140)
    assert blocks[0].confidence == 0.8  # weakest part


def test_does_not_group_large_gap():
    lines = ocr.group_lines([tb("A", 0, 100, 50, 140), tb("B", 200, 100, 250, 140)])
    assert len(lines) == 2


def test_does_not_group_different_heights():
    lines = ocr.group_lines([tb("BIG", 0, 100, 100, 200), tb("small", 110, 180, 160, 200)])
    assert len(lines) == 2


def test_does_not_group_different_baselines():
    lines = ocr.group_lines([tb("UP", 0, 100, 50, 140), tb("DOWN", 60, 130, 110, 170)])
    assert len(lines) == 2


def test_groups_three_words_and_keeps_separate_lines():
    blocks = [
        tb("CALLE", 0, 0, 100, 30), tb("MAYOR", 110, 0, 210, 30), tb("12", 220, 2, 260, 31),
        tb("ABIERTO", 0, 100, 140, 130),
    ]
    texts = sorted(" ".join(b.text for b in line) for line in ocr.group_lines(blocks))
    assert texts == ["ABIERTO", "CALLE MAYOR 12"]


# --- read_text ---------------------------------------------------------------------------

def test_read_text_uses_cached_reader_per_language_set(monkeypatch):
    created = []

    def factory(langs):
        backend = FakeBackend(langs, [(quad(10, 10, 200, 60), "HOTEL", 0.9)])
        created.append(backend)
        return backend

    monkeypatch.setattr(ocr, "_backend_factory", factory)
    r1 = ocr.read_text(png(), None)
    r2 = ocr.read_text(png(), list(reversed(ocr.DEFAULT_LANGUAGES)))
    ocr.read_text(png(), ["ru", "en"])
    assert len(created) == 2
    assert r1.languages == ocr.DEFAULT_LANGUAGES
    assert r2.blocks[0].text == "HOTEL"
    assert r1.image_size == (1000, 500)
    assert not r1.truncated and r1.total_blocks == 1
    assert ocr.is_loaded(["en", "ru"])


def test_read_text_downscales_only_large_images(monkeypatch):
    backend = FakeBackend([])
    monkeypatch.setattr(ocr, "_backend_factory", lambda langs: backend)
    ocr.read_text(png(2500, 1000))
    assert backend.seen_size == (2500, 1000)
    result = ocr.read_text(png(6000, 3000))
    assert backend.seen_size == (3000, 1500)
    assert result.image_size == (6000, 3000)


def test_read_text_bbox_scaled_back_to_original_pixels(monkeypatch):
    backend = FakeBackend([], [(quad(100, 200, 300, 250), "SIGN", 0.9)])
    monkeypatch.setattr(ocr, "_backend_factory", lambda langs: backend)
    result = ocr.read_text(png(6000, 3000))
    assert result.blocks[0].bbox == (200, 400, 600, 500)
    assert result.region is None and result.source_pixels is None


def test_read_text_region_crops_at_full_resolution(monkeypatch):
    backend = FakeBackend([], [(quad(10, 20, 110, 60), "VIA ROMA", 0.9)])
    monkeypatch.setattr(ocr, "_backend_factory", lambda langs: backend)
    result = ocr.read_text(png(1000, 500), region=[0.5, 0.5, 1.0, 1.0])
    assert backend.seen_size == (500, 250)  # only the crop is read, at native resolution
    assert result.image_size == (1000, 500)
    assert result.region == (0.5, 0.5, 1.0, 1.0)
    assert result.source_pixels == (500, 250, 1000, 500)
    assert result.scale == 1.0 and result.upscaled is False and result.retried is False
    block = result.blocks[0]
    assert block.bbox == (510, 270, 610, 310)  # offset into original pixels
    assert block.relative_size == pytest.approx(100 * 40 / (500 * 250))  # share of the crop


def test_read_text_large_region_is_downscaled_and_mapped_back(monkeypatch):
    backend = FakeBackend([], [(quad(100, 100, 200, 150), "X", 0.9)])
    monkeypatch.setattr(ocr, "_backend_factory", lambda langs: backend)
    # Region 0.25-1 of a 8000x4000 image is 6000x3000, downscaled to 3000x1500.
    result = ocr.read_text(png(8000, 4000), region=[0.25, 0.25, 1.0, 1.0])
    assert backend.seen_size == (3000, 1500)
    assert result.source_pixels == (2000, 1000, 8000, 4000)
    assert result.blocks[0].bbox == (2200, 1200, 2400, 1300)


def test_read_text_rotation_is_opt_in(monkeypatch):
    backend = FakeBackend([], [(quad(0, 0, 40, 40), "7", 0.9)])
    monkeypatch.setattr(ocr, "_backend_factory", lambda langs: backend)
    assert ocr.read_text(png()).rotated_text is False
    assert backend.rotated is False
    assert ocr.read_text(png(), rotated_text=True).rotated_text is True
    assert backend.rotated is True


class EmptyFirstBackend(FakeBackend):
    """Finds nothing on the first read, then returns its detections."""

    def __init__(self, languages, detections=()):
        super().__init__(languages, detections)
        self.sizes = []

    def read(self, image, rotated=False):
        self.sizes.append(image.size)
        return [] if len(self.sizes) == 1 else self.detections


def test_read_text_retries_enlarged_when_nothing_found(monkeypatch):
    backend = EmptyFirstBackend([], [(quad(40, 40, 80, 60), "12", 0.9)])
    monkeypatch.setattr(ocr, "_backend_factory", lambda langs: backend)
    result = ocr.read_text(png(1000, 500), region=[0.0, 0.0, 0.1, 0.1])  # 100x50
    assert backend.sizes == [(100, 50), (400, 200)]  # native, then x4 (the cap)
    assert result.retried is True and result.upscaled is True and result.scale == 4.0
    assert result.blocks[0].bbox == (10, 10, 20, 15)


def test_read_text_retry_scale(monkeypatch):
    backend = EmptyFirstBackend([], [(quad(0, 0, 40, 40), "7", 0.9)])
    monkeypatch.setattr(ocr, "_backend_factory", lambda langs: backend)
    ocr.read_text(png(1000, 500))  # towards 1568 px, but at least x2
    assert backend.sizes[1] == (2000, 1000)
    backend.sizes.clear()
    ocr.read_text(png(2000, 1000))  # x2 would pass MAX_SIDE: capped at 3000 (x1.5)
    assert backend.sizes[1] == (3000, 1500)


def test_read_text_retry_keeps_empty_result(monkeypatch):
    backend = FakeBackend([], [])
    monkeypatch.setattr(ocr, "_backend_factory", lambda langs: backend)
    result = ocr.read_text(png(1000, 500), region=[0.0, 0.0, 0.1, 0.1])
    assert result.retried is True and result.blocks == [] and result.scale == 1.0


def test_read_text_no_retry_when_text_found_or_no_room(monkeypatch):
    backend = EmptyFirstBackend([], [(quad(0, 0, 40, 40), "7", 0.9)])
    monkeypatch.setattr(ocr, "_backend_factory", lambda langs: backend)
    assert ocr.read_text(png(4000, 2000)).retried is False  # above MAX_SIDE: no room to enlarge
    assert len(backend.sizes) == 1

    backend = FakeBackend([], [(quad(0, 0, 40, 40), "7", 0.9)])
    monkeypatch.setattr(ocr, "_backend_factory", lambda langs: backend)
    monkeypatch.setattr(ocr, "_readers", {})
    assert ocr.read_text(png(1000, 500), region=[0, 0, 0.1, 0.1]).retried is False


@pytest.mark.parametrize("region", [[0.5, 0.5, 0.4, 1.0], [0, 0, 1.5, 1], [0, 0, 0.01, 1]])
def test_read_text_invalid_region(monkeypatch, region):
    monkeypatch.setattr(ocr, "_backend_factory", lambda langs: FakeBackend(langs))
    with pytest.raises(OCRError):
        ocr.read_text(png(), region=region)


def test_read_text_rejects_bad_image(monkeypatch):
    monkeypatch.setattr(ocr, "_backend_factory", lambda langs: FakeBackend(langs))
    with pytest.raises(ValueError):
        ocr.read_text(b"not an image")


def test_easyocr_backend_without_extra(monkeypatch):
    import builtins

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "easyocr":
            raise ImportError("no easyocr")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    with pytest.raises(ocr.OCRUnavailableError, match=r"geotrace\[ocr\]"):
        ocr.EasyOCRBackend(["en"])
