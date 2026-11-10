"""The photographic-style year head, and how time_from_objects reports it.

The head is a plain linear layer over a frozen OpenCLIP embedding, so its arithmetic is checked
directly on synthetic logits: no backbone, no weights download, no torch.
"""
import pytest

pytest.importorskip("numpy")

import numpy as np

from geotrace.core import photo_dating as pd
from geotrace.tools import time_from_objects as tfo

YEARS = np.arange(1950, 2000, dtype=np.int32)
IDENTITY = np.eye(50, dtype=np.float32)
NO_BIAS = np.zeros(50, dtype=np.float32)


def logits_peaking_at(year, sharpness=12.0):
    """An embedding that, through the identity weight, becomes logits peaked on one year."""
    return (sharpness * np.exp(-0.5 * ((YEARS - year) / 2.0) ** 2)).astype(np.float32)


def test_argmax_year_is_reported():
    out = pd.predict_from_embedding(logits_peaking_at(1980), IDENTITY, NO_BIAS, YEARS)
    assert out["estimated_year"] == 1980
    assert out["model_year_range"] == [1950, 1999]
    assert 0.0 < out["confidence"] <= 1.0


def test_expected_year_tracks_a_symmetric_peak():
    out = pd.predict_from_embedding(logits_peaking_at(1975), IDENTITY, NO_BIAS, YEARS)
    assert out["expected_year"] == pytest.approx(1975, abs=0.5)


def test_interval_brackets_the_estimate_and_widens_when_unsure():
    sharp = pd.predict_from_embedding(logits_peaking_at(1980, 20.0), IDENTITY, NO_BIAS, YEARS)
    vague = pd.predict_from_embedding(logits_peaking_at(1980, 0.5), IDENTITY, NO_BIAS, YEARS)
    lo, hi = sharp["interval_80pct"]
    assert lo <= 1980 <= hi
    sharp_width = sharp["interval_80pct"][1] - sharp["interval_80pct"][0]
    vague_width = vague["interval_80pct"][1] - vague["interval_80pct"][0]
    assert vague_width > sharp_width
    assert vague["confidence"] < sharp["confidence"]


def test_uniform_probabilities_give_an_80_percent_interval():
    flat = np.ones(50, dtype=np.float32) / 50
    lo, hi = pd._interval(YEARS, flat, 0.8)
    # Every window of the same width is equally good, so the first one wins; 40 years carries
    # exactly 0.8, which float32 accumulation can leave a hair short, so 41 is also correct.
    assert lo == 1950
    assert (hi - lo + 1) in (40, 41)


def test_mae_is_passed_through():
    out = pd.predict_from_embedding(logits_peaking_at(1960), IDENTITY, NO_BIAS, YEARS, mae=6.4)
    assert out["model_mae_years"] == 6.4
    assert "1950-1999" in out["out_of_range_warning"]


class FakeContext:
    def __init__(self):
        self.messages = []

    async def info(self, message):
        self.messages.append(message)


INTERVALS = {
    "current_year": 2026,
    "image_taken_not_before_year": 1975,
    "objects": [{"object": "bicycle", "built_from_year": 1975, "built_until_year": 2026,
                 "tolerance_years": 5.0}],
}
DATING = {"estimated_year": 1982, "expected_year": 1983.4, "interval_80pct": [1974, 1991],
          "confidence": 0.12, "model_year_range": [1950, 1999]}


@pytest.fixture
def stub(monkeypatch):
    async def fake_load(image):
        return b"jpeg"

    monkeypatch.setattr(tfo, "load_image", fake_load)
    monkeypatch.setattr(tfo, "estimate_intervals", lambda data, objects: dict(INTERVALS))
    monkeypatch.setattr(tfo, "estimate_capture_year", lambda data: dict(DATING))


async def run():
    return await tfo.time_from_objects(image="photo.jpg", objects=["bicycle"], ctx=FakeContext())  # type: ignore[arg-type]


async def test_tool_reports_both_signals(stub):
    out = await run()
    assert out["image_taken_not_before_year"] == 1975      # the object interval survives
    assert out["capture_year_estimate"]["estimated_year"] == 1982
    assert "capture_year_error" not in out


async def test_tool_still_answers_when_the_style_model_fails(monkeypatch, stub):
    def boom(data):
        raise ImportError("No module named 'open_clip'")

    monkeypatch.setattr(tfo, "estimate_capture_year", boom)
    out = await run()
    assert out["image_taken_not_before_year"] == 1975      # object intervals are unaffected
    assert "open_clip" in out["capture_year_error"]
    assert "capture_year_estimate" not in out


async def test_tool_flags_an_object_newer_than_the_classifier_can_express(monkeypatch, stub):
    monkeypatch.setattr(
        tfo, "estimate_intervals",
        lambda data, objects: {**INTERVALS, "image_taken_not_before_year": 2015},
    )
    out = await run()
    assert "2015" in out["capture_year_estimate"]["disagrees_with_objects"]


async def test_tool_does_not_flag_a_consistent_pair(stub):
    out = await run()
    assert "disagrees_with_objects" not in out["capture_year_estimate"]
