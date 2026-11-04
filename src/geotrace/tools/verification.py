from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Annotated

import httpx
from mcp.server.mcpserver import Context
from mcp.server.mcpserver.exceptions import ToolError
from pydantic import BaseModel, Field

from geotrace.core import extras, images, mapillary, matching, retrieval
from geotrace.server import mcp

MAX_CANDIDATES = 50
MIN_RADIUS_M = 25
MAX_RADIUS_M = 500
DEFAULT_RADIUS_M = 150
BATCH_SIZE = 10
REFS_PER_CANDIDATE = 40      # nearest diverse Mapillary photos kept per candidate
FINALISTS_PER_BATCH = 20     # top MegaLoc results verified with LightGlue in each batch


class CandidatePoint(BaseModel):
    latitude: float = Field(ge=-90, le=90, description="Latitude in decimal degrees.")
    longitude: float = Field(ge=-180, le=180, description="Longitude in decimal degrees.")


@dataclass
class PlaceMatch:
    candidate_index: int
    reference_id: str
    reference_location: tuple[float, float]
    reference_heading_deg: float | None
    reference_date: str | None        # ISO 8601, converted from captured_at (ms since epoch)
    inliers: int
    verdict: str                      # "match" | "similar" | "no_match"


@dataclass
class VerificationResult:
    matches: list[PlaceMatch]                  # only "match" and "similar", best first
    candidates_checked: int
    candidates_without_coverage: list[int]     # no street-level photos: NOT ruled out
    stopped_early: bool


def _iso_date(captured_at_ms: int | None) -> str | None:
    if captured_at_ms is None:
        return None
    return datetime.fromtimestamp(captured_at_ms / 1000, tz=UTC).isoformat()


def _require_extra() -> None:
    try:
        extras.require_verification_extra()
    except extras.MissingExtraError as exc:
        raise ToolError(str(exc)) from exc


async def _load(source: str, what: str) -> bytes:
    try:
        return await images.load_image(source)
    except (ValueError, OSError, httpx.HTTPError) as exc:
        raise ToolError(f"Could not load the {what} ({source}): {exc}") from exc


@mcp.tool(
    name="verify_candidate_locations",
    description="Checks candidate locations for a street-level photo against nearby Mapillary "
    "photos. For each candidate it takes the Mapillary photos within radius_m, ranks them by visual "
    "similarity with MegaLoc, and verifies the best ones geometrically with LightGlue and RANSAC. "
    "The images never pass through the model. "
    "Sort candidates from most to least likely: they are processed in that order, in batches of "
    f"{BATCH_SIZE}, and the search stops at the first 'match' (stopped_early=true). "
    f"At most {MAX_CANDIDATES} candidates; radius_m from {MIN_RADIUS_M} to {MAX_RADIUS_M} (default "
    f"{DEFAULT_RADIUS_M}). "
    "Returns matches (best first, each with candidate_index into your list, the Mapillary "
    "reference_id, reference_location (lat, lon), reference_heading_deg, reference_date and inliers), "
    "candidates_checked, candidates_without_coverage and stopped_early. "
    f"verdict 'match' ({matching.MATCH_MIN_INLIERS}+ RANSAC inliers) means the same place, "
    "geometrically verified. 'similar' "
    f"({matching.SIMILAR_MIN_INLIERS}-{matching.MATCH_MIN_INLIERS - 1} inliers) is a weak lead to "
    "check by other means, not a confirmation. A candidate without a match is unlikely but not "
    "excluded: the photo may face another way or the place may have changed. "
    "A candidate in candidates_without_coverage has no Mapillary photos nearby: it was NOT ruled "
    "out, it just could not be checked. Requires the MAPILLARY_TOKEN environment variable.",
)
async def verify_candidate_locations(
    image: images.ImageSource,
    candidates: Annotated[
        list[CandidatePoint],
        Field(description="Candidate points, sorted from most to least likely."),
    ],
    ctx: Context,
    radius_m: Annotated[
        float,
        Field(description=f"Search radius around each candidate, in metres "
              f"({MIN_RADIUS_M}-{MAX_RADIUS_M}). Default {DEFAULT_RADIUS_M}."),
    ] = DEFAULT_RADIUS_M,
) -> VerificationResult:
    if not candidates:
        raise ToolError("candidates must contain at least one point.")
    if len(candidates) > MAX_CANDIDATES:
        raise ToolError(
            f"Too many candidates ({len(candidates)}); the maximum is {MAX_CANDIDATES}. "
            f"Send the {MAX_CANDIDATES} most likely first."
        )
    if not MIN_RADIUS_M <= radius_m <= MAX_RADIUS_M:
        raise ToolError(f"radius_m must be between {MIN_RADIUS_M} and {MAX_RADIUS_M} metres.")
    _require_extra()
    try:
        token = mapillary.get_token()
    except mapillary.MapillaryError as exc:
        raise ToolError(str(exc)) from exc
    query = await _load(image, "image")

    points = [(c.latitude, c.longitude) for c in candidates]
    total = len(points)
    seen_ids: set[str] = set()
    found: list[PlaceMatch] = []
    without_coverage: list[int] = []
    checked = 0
    stopped_early = False

    for start in range(0, total, BATCH_SIZE):
        batch = points[start:start + BATCH_SIZE]
        end = start + len(batch)
        await ctx.info(f"Candidates {start + 1}-{end} of {total}: searching Mapillary")
        try:
            refs, no_cov = await mapillary.candidate_image_ids(
                batch, token, radius_m, REFS_PER_CANDIDATE, index_offset=start, seen_ids=seen_ids,
            )
        except mapillary.MapillaryError as exc:
            raise ToolError(str(exc)) from exc
        without_coverage.extend(no_cov)

        if refs:
            await ctx.info(f"Ranking {len(refs)} Mapillary photos with MegaLoc")
            try:
                ranked = await retrieval.rank_references(query, refs, top_k=FINALISTS_PER_BATCH)
            except ValueError as exc:
                raise ToolError(f"Could not process the image: {exc}") from exc
            finalists = [r.image for r in ranked]
            thumbs = await mapillary.download_thumbnails(finalists, 1024)
            await ctx.info(f"Verifying the {len(finalists)} most similar photos with LightGlue")
            for ref in finalists:
                data = thumbs.get(ref.id)
                if data is None:
                    continue
                try:
                    pair = await matching.verify_pair(query, data)
                except ValueError:
                    continue                    # unreadable reference thumbnail
                if pair.verdict == "no_match":
                    continue
                found.append(PlaceMatch(
                    candidate_index=ref.candidate_index,
                    reference_id=ref.id,
                    reference_location=(ref.lat, ref.lon),
                    reference_heading_deg=ref.compass_angle,
                    reference_date=_iso_date(ref.captured_at),
                    inliers=pair.inliers,
                    verdict=pair.verdict,
                ))
                if pair.verdict == "match":
                    stopped_early = True
                    break

        checked = end
        await ctx.report_progress(end, total)
        if stopped_early:
            best = next(m for m in found if m.verdict == "match")
            await ctx.info(f"Match at candidate {best.candidate_index} ({best.inliers} inliers); stopping")
            break

    found.sort(key=lambda m: (m.verdict != "match", -m.inliers))
    return VerificationResult(
        matches=found,
        candidates_checked=checked,
        candidates_without_coverage=without_coverage,
        stopped_early=stopped_early,
    )


@mcp.tool(
    name="compare_images",
    description="Checks whether two images show the same place, by geometric verification only: "
    "LightGlue feature matches filtered with RANSAC. The images never pass through the model. "
    f"Returns inliers, matches and verdict: 'match' ({matching.MATCH_MIN_INLIERS}+ inliers) means "
    "the same place, geometrically verified; 'similar' "
    f"({matching.SIMILAR_MIN_INLIERS}-{matching.MATCH_MIN_INLIERS - 1}) is a weak lead, not a "
    "confirmation; 'no_match' means no consistent geometry was found, which can also happen for the "
    "same place seen from very different viewpoints, seasons or times of day.",
)
async def compare_images(
    image: images.ImageSource,
    reference: Annotated[str, Field(description="URL or local path of the reference image.")],
    ctx: Context,
) -> matching.PairMatch:
    _require_extra()
    query = await _load(image, "image")
    ref = await _load(reference, "reference image")
    await ctx.info("Matching the two images with LightGlue")
    try:
        result = await matching.verify_pair(query, ref)
    except ValueError as exc:
        raise ToolError(str(exc)) from exc
    await ctx.info(f"{result.inliers} inliers: {result.verdict}")
    return result
