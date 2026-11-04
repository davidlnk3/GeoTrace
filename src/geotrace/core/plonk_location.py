"""Where an image was probably taken, using PLONK, as regions with probabilities.

PLONK ("Around the World in 80 Timesteps", arXiv:2412.06781, MIT licence, code at
github.com/nicolas-dufour/plonk, pip package diff-plonk) generates coordinates on the sphere
by denoising random noise, conditioned on the image. This module draws many samples and
summarises them: clusters of samples become regions, and the rest of the description is
computed from the same samples. The rules are this project's, not the paper's, and are
described in the docstrings below.
"""
import contextlib
import io
import sys
import threading
from collections import Counter
from functools import lru_cache

import numpy as np
from PIL import Image, ImageOps

MODEL_ID = "nicolas-dufour/PLONK_YFCC"   # trained on YFCC-100M, uses a DINOv2-L encoder
DEFAULT_SAMPLES = 100                    # samples drawn per image; more is slower and steadier

BANDWIDTH_KM = 100.0     # samples within this distance of a seed form one region
MAX_REGIONS = 3
MIN_SHARE = 0.05         # a region needs at least this share of the samples
COVERAGE = 0.90          # area_90_km2 is the area that holds this share of the samples
EARTH_RADIUS_KM = 6371.0

_lock = threading.Lock()


@lru_cache(maxsize=1)
def _pipeline():
    from plonk import PlonkPipeline

    return PlonkPipeline(MODEL_ID)


def _load_pipeline():
    # Loading is not thread-safe, and the tool runs in a worker thread.
    with _lock:
        return _pipeline()


def _open_rgb(image_bytes: bytes) -> Image.Image:
    with Image.open(io.BytesIO(image_bytes)) as img:
        return ImageOps.exif_transpose(img).convert("RGB")


def _haversine_km(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 2 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def _mean_location(lat, lon):
    """Centre of a set of points on the sphere: normalised mean of their 3-D vectors."""
    la, lo = np.radians(lat), np.radians(lon)
    v = np.stack([np.cos(la) * np.cos(lo), np.cos(la) * np.sin(lo), np.sin(la)], axis=1).mean(0)
    v /= np.linalg.norm(v)
    return float(np.degrees(np.arcsin(v[2]))), float(np.degrees(np.arctan2(v[1], v[0])))


def _regions(lat, lon):
    """Greedy clustering. Each region is the samples within BANDWIDTH_KM of the sample with the
    most neighbours among those not yet assigned. Its centre is the mean of its members, and its
    radius is the 90th percentile of the members' distances from the centre."""
    n = len(lat)
    min_members = max(1, int(np.ceil(MIN_SHARE * n)))
    remaining = np.ones(n, dtype=bool)
    regions = []
    for _ in range(MAX_REGIONS):
        idx = np.flatnonzero(remaining)
        if len(idx) < min_members:
            break
        d = _haversine_km(lat[idx][:, None], lon[idx][:, None], lat[idx][None, :], lon[idx][None, :])
        neighbours = (d <= BANDWIDTH_KM).sum(axis=1)
        if neighbours.max() < min_members:
            break
        seed = idx[int(np.argmax(neighbours))]
        members = idx[_haversine_km(lat[seed], lon[seed], lat[idx], lon[idx]) <= BANDWIDTH_KM]
        c_lat, c_lon = _mean_location(lat[members], lon[members])
        dist = _haversine_km(c_lat, c_lon, lat[members], lon[members])
        regions.append({
            "center_lat": c_lat,
            "center_lon": c_lon,
            "radius_km": max(1.0, float(np.percentile(dist, 90))),
            "count": int(len(members)),
        })
        remaining[members] = False
    return regions, float(remaining.sum()) / n


def _area_km2_of_cell(lat_deg):
    """Area of a 1° x 1° cell whose southern edge is at lat_deg (km²)."""
    r = np.radians
    return EARTH_RADIUS_KM ** 2 * r(1.0) * (np.sin(r(lat_deg + 1)) - np.sin(r(lat_deg)))


def _area_for_coverage(lat, lon):
    """Area of the fewest 1° cells that hold COVERAGE of the samples."""
    cells = Counter(zip(np.floor(lat).astype(int), np.floor(lon).astype(int)))
    target = COVERAGE * len(lat)
    covered, area = 0, 0.0
    for (cell_lat, _), count in cells.most_common():
        area += float(_area_km2_of_cell(cell_lat))
        covered += count
        if covered >= target:
            break
    return area


def _localizability(area_km2):
    if area_km2 <= 10_000:
        return "high"
    if area_km2 <= 500_000:
        return "medium"
    return "low"


def describe_location(image_bytes: bytes, num_samples: int = DEFAULT_SAMPLES) -> dict:
    """Return the regions where the image was probably taken, with their probabilities.

    regions are sorted by probability; regions[0] is the most probable area and its centre is
    the most probable point. Each region's name is the nearest populated place and its admin1
    region, from reverse_geocoder (offline, GeoNames data). by_country gives each country's
    share of all samples; unassigned_probability is the share not in any returned region.
    """
    import reverse_geocoder as rg

    if num_samples < 1:
        raise ValueError("num_samples must be at least 1.")

    pipe = _load_pipeline()
    img = _open_rgb(image_bytes)

    # PLONK and reverse_geocoder print progress to stdout, which would corrupt the MCP stdio protocol.
    with contextlib.redirect_stdout(sys.stderr):
        coords = np.asarray(pipe(img, batch_size=num_samples), dtype=float)   # (N, 2) lat, lon
        lat, lon = coords[:, 0], coords[:, 1]
        regions, unassigned = _regions(lat, lon)
        sample_places = rg.search(list(zip(lat.tolist(), lon.tolist())), mode=1)
        region_places = rg.search([(r["center_lat"], r["center_lon"]) for r in regions], mode=1)

    countries = Counter(p["cc"].lower() for p in sample_places)
    by_country = {cc: round(c / num_samples, 3) for cc, c in countries.most_common()}
    area = _area_for_coverage(lat, lon)

    out_regions = []
    for region, place in zip(regions, region_places):
        out_regions.append({
            "name": f"{place['name']}, {place['admin1']}",
            "probability": round(region["count"] / num_samples, 3),
            "center": {"latitude": round(region["center_lat"], 5),
                       "longitude": round(region["center_lon"], 5)},
            "radius_km": round(region["radius_km"], 1),
            "country_code": place["cc"].lower(),
        })

    return {
        "model": MODEL_ID,
        "num_samples": num_samples,
        "regions": out_regions,
        "unassigned_probability": round(unassigned, 3),
        "by_country": by_country,
        "area_90_km2": round(area),
        "localizability": _localizability(area),
    }
