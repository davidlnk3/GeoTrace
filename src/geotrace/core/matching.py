"""Geometric verification of an image pair with LightGlue and RANSAC.

Local features come from ALIKED (BSD-3-Clause) or DISK (Apache-2.0), matched with LightGlue
(https://github.com/cvg/LightGlue, Apache-2.0). SuperPoint is deliberately not used: its weights
are licensed for non-commercial use only, and GeoTrace is MIT.

The matches are then filtered with a RANSAC fundamental matrix (MAGSAC++), which fits any static
3-D scene seen from two viewpoints. The number of inliers decides the verdict.
"""

import asyncio
import hashlib
import threading
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, Literal

from geotrace.core.images import open_rgb

# Thresholds on RANSAC inliers. Not tuned on a benchmark yet.
MATCH_MIN_INLIERS = 60
SIMILAR_MIN_INLIERS = 20

EXTRACTOR: Literal["aliked", "disk"] = "aliked"   # ALIKED is lighter than DISK on a CPU
MAX_KEYPOINTS = 2048
MAX_SIDE = 1024                                    # images are matched at most at this size
RANSAC_THRESHOLD_PX = 2.0
RANSAC_CONFIDENCE = 0.999
RANSAC_MAX_ITERS = 10_000
MIN_MATCHES_FOR_RANSAC = 8                         # the fundamental matrix needs 8 points
FEATURE_CACHE_ENTRIES = 8                          # the query is reused across many pairs

Verdict = Literal["match", "similar", "no_match"]


@dataclass(frozen=True)
class PairMatch:
    inliers: int              # matches consistent with one rigid two-view geometry
    matches: int              # LightGlue matches before RANSAC
    verdict: Verdict          # "match" (same place), "similar", or "no_match"


def verdict_for(inliers: int) -> Verdict:
    if inliers >= MATCH_MIN_INLIERS:
        return "match"
    if inliers >= SIMILAR_MIN_INLIERS:
        return "similar"
    return "no_match"


_lock = threading.Lock()
_models: tuple[Any, Any] | None = None
_feature_cache: OrderedDict[str, dict] = OrderedDict()


def _load_models() -> tuple[Any, Any]:
    """Load the extractor and LightGlue once. Weights are downloaded on first use."""
    global _models
    with _lock:
        if _models is None:
            from lightglue import ALIKED, DISK, LightGlue

            extractor_cls = ALIKED if EXTRACTOR == "aliked" else DISK
            extractor = extractor_cls(max_num_keypoints=MAX_KEYPOINTS).eval()
            matcher = LightGlue(features=EXTRACTOR).eval()
            _models = (extractor, matcher)
        return _models


def _features(extractor: Any, data: bytes) -> dict:
    """Keypoints and descriptors for one image, cached by content hash."""
    import numpy as np
    import torch

    key = hashlib.sha256(data).hexdigest()
    with _lock:
        cached = _feature_cache.get(key)
        if cached is not None:
            _feature_cache.move_to_end(key)
            return cached
    rgb = np.asarray(open_rgb(data, MAX_SIDE), dtype=np.float32) / 255.0
    tensor = torch.from_numpy(rgb).permute(2, 0, 1).contiguous()
    with torch.inference_mode():
        feats = extractor.extract(tensor, resize=None)
    with _lock:
        _feature_cache[key] = feats
        while len(_feature_cache) > FEATURE_CACHE_ENTRIES:
            _feature_cache.popitem(last=False)
    return feats


def _count_inliers(points0: Any, points1: Any) -> int:
    import cv2
    import numpy as np

    if len(points0) < MIN_MATCHES_FOR_RANSAC:
        return 0
    _, mask = cv2.findFundamentalMat(
        np.ascontiguousarray(points0, dtype=np.float64),
        np.ascontiguousarray(points1, dtype=np.float64),
        cv2.USAC_MAGSAC,
        RANSAC_THRESHOLD_PX,
        RANSAC_CONFIDENCE,
        RANSAC_MAX_ITERS,
    )
    return 0 if mask is None else int(mask.sum())


def verify_pair_sync(query_bytes: bytes, reference_bytes: bytes) -> PairMatch:
    """Blocking version of verify_pair."""
    import torch
    from lightglue.utils import rbd

    extractor, matcher = _load_models()
    feats0 = _features(extractor, query_bytes)
    feats1 = _features(extractor, reference_bytes)
    with torch.inference_mode():
        out = matcher({"image0": feats0, "image1": feats1})
    f0, f1, m01 = rbd(feats0), rbd(feats1), rbd(out)
    matches = m01["matches"]
    points0 = f0["keypoints"][matches[:, 0]].cpu().numpy()
    points1 = f1["keypoints"][matches[:, 1]].cpu().numpy()
    inliers = _count_inliers(points0, points1)
    return PairMatch(inliers=inliers, matches=int(matches.shape[0]), verdict=verdict_for(inliers))


async def verify_pair(query_bytes: bytes, reference_bytes: bytes) -> PairMatch:
    """Check whether two images show the same place, by LightGlue matches verified with RANSAC.

    Raises ValueError if either image cannot be decoded.
    """
    try:
        return await asyncio.to_thread(verify_pair_sync, query_bytes, reference_bytes)
    except (OSError, SyntaxError) as exc:   # what Pillow raises for unreadable images
        raise ValueError(f"Could not decode an image: {exc}") from exc
