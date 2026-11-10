"""Selection logic of the merged relative_sun_position tool.

The shadow stage (SSIS on a Space + GeoCalib) and the DINOv2 lighting model are both replaced, so
these tests cover only the choice between them: a shadow pair above the confidence threshold wins,
otherwise the lighting model answers.
"""
import pytest

pytest.importorskip("numpy")

from geotrace.tools import relative_sun_position as rsp

LIGHTING = {"sun_altitude_deg": 12.5, "sun_azimuth_relative_camera_deg": 200.0}


class FakeContext:
    def __init__(self):
        self.messages = []

    async def info(self, message):
        self.messages.append(message)


def shadows(*scores):
    """A shadow stage result with one pair per score, best first (as the tool sorts them)."""
    ordered = sorted(scores, reverse=True)
    return {
        "num_pairs": len(ordered),
        "pairs": [
            {
                "pair_score": s,
                "sun_altitude_deg": 40.0 + i,
                "sun_relative_azimuth_to_camera_deg": 100.0 + i,
            }
            for i, s in enumerate(ordered)
        ],
        "skipped": [{"pair_id": 7, "reason": "tip above the horizon"}],
    }


@pytest.fixture(autouse=True)
def stub_stages(monkeypatch):
    async def fake_load(image):
        return b"jpeg"

    monkeypatch.setattr(rsp, "load_image", fake_load)
    monkeypatch.setattr(rsp, "estimate", lambda data: dict(LIGHTING))


def use_shadows(monkeypatch, result=None, error=None):
    async def fake(image, ctx):
        if error is not None:
            raise error
        return result

    monkeypatch.setattr(rsp, "_from_shadows", fake)


async def run():
    return await rsp.relative_sun_position(image="photo.jpg", ctx=FakeContext())  # type: ignore[arg-type]


async def test_confident_pair_wins(monkeypatch):
    use_shadows(monkeypatch, shadows(0.91))
    out = await run()
    assert out["source"] == "shadows"
    assert out["pair_score"] == 0.91
    assert out["sun_altitude_deg"] == 40.0
    assert out["sun_relative_azimuth_to_camera_deg"] == 100.0
    assert out["num_pairs"] == 1 and out["skipped"]  # full detail kept


async def test_highest_scoring_pair_is_used(monkeypatch):
    use_shadows(monkeypatch, shadows(0.85, 0.95, 0.82))
    out = await run()
    assert out["source"] == "shadows"
    assert out["pair_score"] == 0.95
    assert out["sun_altitude_deg"] == 40.0  # the 0.95 pair sorts first
    assert out["num_pairs"] == 3


async def test_falls_back_when_no_pair_is_confident(monkeypatch):
    use_shadows(monkeypatch, shadows(0.75, 0.6))
    out = await run()
    assert out["source"] == "outdoor_lighting"
    assert out["sun_altitude_deg"] == 12.5
    assert out["sun_relative_azimuth_to_camera_deg"] == 200.0  # renamed from the model's field
    assert "pair_score" not in out
    assert out["num_pairs"] == 2 and out["pairs"]  # shadows were found, just not trusted


async def test_threshold_is_exclusive(monkeypatch):
    use_shadows(monkeypatch, shadows(rsp.SHADOW_CONFIDENCE))
    out = await run()
    assert out["source"] == "outdoor_lighting"


async def test_falls_back_when_no_pairs_found(monkeypatch):
    use_shadows(monkeypatch, {"num_pairs": 0, "pairs": [], "skipped": []})
    out = await run()
    assert out["source"] == "outdoor_lighting"
    assert out["num_pairs"] == 0
    assert "shadow_error" not in out


async def test_falls_back_when_shadow_stage_unavailable(monkeypatch):
    use_shadows(monkeypatch, error=ImportError("No module named 'geocalib'"))
    out = await run()
    assert out["source"] == "outdoor_lighting"
    assert out["sun_altitude_deg"] == 12.5
    assert "geocalib" in out["shadow_error"]
    assert "pairs" not in out  # the stage never produced any
