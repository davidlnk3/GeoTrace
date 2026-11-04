"""Tests for tools/verification.py with core.mapillary, core.retrieval and core.matching mocked."""

from collections.abc import Sequence

import pytest
from mcp.server.mcpserver.exceptions import ToolError

from geotrace.core import extras, mapillary, matching, retrieval
from geotrace.core.mapillary import MapillaryImage
from geotrace.core.matching import PairMatch
from geotrace.core.retrieval import RankedReference
from geotrace.tools import verification
from geotrace.tools.verification import CandidatePoint, verify_candidate_locations


class FakeContext:
    def __init__(self):
        self.progress: list[tuple[float, float | None]] = []
        self.messages: list[str] = []

    async def info(self, data, **_):
        self.messages.append(str(data))

    async def report_progress(self, progress, total=None, message=None):
        self.progress.append((progress, total))


def candidates(n: int) -> list[CandidatePoint]:
    return [CandidatePoint(latitude=40 + i * 0.01, longitude=-3) for i in range(n)]


@pytest.fixture
def pipeline(monkeypatch):
    """Mock every external step. `state["match_at"]` is the candidate index whose photo matches."""
    state: dict = {"match_at": None, "similar_at": set(), "no_coverage": set(), "searched": []}

    monkeypatch.setenv("MAPILLARY_TOKEN", "MLY|test")
    monkeypatch.setattr(extras, "require_verification_extra", lambda: None)

    async def fake_load_image(source: str) -> bytes:
        return source.encode() if source.startswith("ref") else b"query"

    async def fake_candidate_image_ids(points, token, radius_m, per_candidate, *, index_offset=0,
                                       seen_ids=None, client=None):
        state["searched"].append((index_offset, len(points)))
        selected, no_cov = [], []
        for i, (lat, lon) in enumerate(points):
            idx = index_offset + i
            if idx in state["no_coverage"]:
                no_cov.append(idx)
                continue
            selected.append(MapillaryImage(
                id=f"ref{idx}", lat=lat, lon=lon, compass_angle=90.0, captured_at=1_600_000_000_000,
                sequence="s", is_pano=False, thumb_256_url="u", thumb_1024_url="u",
                candidate_index=idx,
            ))
        return selected, no_cov

    async def fake_rank(query: bytes, refs: Sequence[MapillaryImage], top_k: int = 20):
        return [RankedReference(image=r, score=1.0) for r in refs][:top_k]

    async def fake_thumbs(images, size, client=None):
        assert size == 1024
        return {i.id: i.id.encode() for i in images}

    async def fake_verify(query: bytes, ref: bytes) -> PairMatch:
        idx = int(ref.decode().removeprefix("ref"))
        if idx == state["match_at"]:
            return PairMatch(inliers=120, matches=150, verdict="match")
        if idx in state["similar_at"]:
            return PairMatch(inliers=30, matches=80, verdict="similar")
        return PairMatch(inliers=3, matches=10, verdict="no_match")

    monkeypatch.setattr(verification.images, "load_image", fake_load_image)
    monkeypatch.setattr(mapillary, "candidate_image_ids", fake_candidate_image_ids)
    monkeypatch.setattr(mapillary, "download_thumbnails", fake_thumbs)
    monkeypatch.setattr(retrieval, "rank_references", fake_rank)
    monkeypatch.setattr(matching, "verify_pair", fake_verify)
    return state


@pytest.mark.anyio
async def test_stops_at_first_match_and_reports_progress(pipeline):
    pipeline["match_at"] = 14            # second batch
    pipeline["similar_at"] = {3}
    ctx = FakeContext()

    result = await verify_candidate_locations("photo.jpg", candidates(35), ctx)  # type: ignore[arg-type]

    assert result.stopped_early is True
    assert result.candidates_checked == 20
    assert pipeline["searched"] == [(0, 10), (10, 10)]          # the third batch never ran
    assert ctx.progress == [(10, 35), (20, 35)]
    assert [(m.candidate_index, m.verdict) for m in result.matches] == [(14, "match"), (3, "similar")]
    best = result.matches[0]
    assert best.reference_id == "ref14"
    assert best.reference_heading_deg == 90.0
    assert best.reference_date == "2020-09-13T12:26:40+00:00"
    assert best.reference_location == (40 + 14 * 0.01, -3)


@pytest.mark.anyio
async def test_checks_everything_without_a_match(pipeline):
    pipeline["no_coverage"] = {2, 11}
    ctx = FakeContext()

    result = await verify_candidate_locations("photo.jpg", candidates(12), ctx)  # type: ignore[arg-type]

    assert result.stopped_early is False
    assert result.candidates_checked == 12
    assert result.matches == []
    assert result.candidates_without_coverage == [2, 11]
    assert ctx.progress == [(10, 12), (12, 12)]
    assert ctx.messages                                          # status messages were sent


@pytest.mark.anyio
async def test_missing_token_raises_tool_error(pipeline, monkeypatch):
    monkeypatch.delenv("MAPILLARY_TOKEN")
    with pytest.raises(ToolError, match="MAPILLARY_TOKEN"):
        await verify_candidate_locations("photo.jpg", candidates(1), FakeContext())  # type: ignore[arg-type]
    assert pipeline["searched"] == []


@pytest.mark.anyio
@pytest.mark.parametrize("n, radius", [(0, 150), (51, 150), (5, 24), (5, 501)])
async def test_input_limits(pipeline, n, radius):
    with pytest.raises(ToolError):
        await verify_candidate_locations(
            "photo.jpg", candidates(n), FakeContext(), radius_m=radius,  # type: ignore[arg-type]
        )


@pytest.mark.anyio
async def test_missing_extra_raises_tool_error(pipeline, monkeypatch):
    def missing():
        raise extras.MissingExtraError("install with uv sync --extra verification")

    monkeypatch.setattr(extras, "require_verification_extra", missing)
    with pytest.raises(ToolError, match="uv sync --extra verification"):
        await verify_candidate_locations("photo.jpg", candidates(1), FakeContext())  # type: ignore[arg-type]


@pytest.mark.anyio
async def test_compare_images_uses_verify_pair(pipeline):
    result = await verification.compare_images("a.jpg", "ref7", FakeContext())  # type: ignore[arg-type]
    assert result.verdict == "no_match"


def test_verdict_thresholds():
    assert matching.verdict_for(60) == "match"
    assert matching.verdict_for(59) == "similar"
    assert matching.verdict_for(20) == "similar"
    assert matching.verdict_for(19) == "no_match"
