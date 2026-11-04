"""Candidate locations for an image, sampled from PLONK.

PLONK ("Around the World in 80 Timesteps", arXiv:2412.06781, MIT licence, code at
github.com/nicolas-dufour/plonk, pip package diff-plonk) generates coordinates on the sphere
by denoising random noise, conditioned on the image. This module only draws those samples;
summarising them into regions is in location_regions.py.
"""
import contextlib
import io
import sys
import threading
from functools import lru_cache

import numpy as np
from PIL import Image, ImageOps

MODEL_ID = "nicolas-dufour/PLONK_YFCC"   # trained on YFCC-100M, uses a DINOv2-L encoder
DEFAULT_SAMPLES = 100                    # samples drawn per image; more is slower and steadier

_lock = threading.Lock()


@lru_cache(maxsize=1)
def _pipeline():
    from plonk import PlonkPipeline

    return PlonkPipeline(MODEL_ID)


def _load_pipeline():
    # Loading is not thread-safe, and the tools run in worker threads.
    with _lock:
        return _pipeline()


def _open_rgb(image_bytes: bytes) -> Image.Image:
    with Image.open(io.BytesIO(image_bytes)) as img:
        return ImageOps.exif_transpose(img).convert("RGB")


def sample_coordinates(image_bytes: bytes, num_samples: int = DEFAULT_SAMPLES):
    """Draw num_samples locations for the image. Returns (latitudes, longitudes) in degrees."""
    if num_samples < 1:
        raise ValueError("num_samples must be at least 1.")

    pipe = _load_pipeline()
    img = _open_rgb(image_bytes)

    # PLONK prints progress to stdout, which would corrupt the MCP stdio protocol.
    with contextlib.redirect_stdout(sys.stderr):
        coords = np.asarray(pipe(img, batch_size=num_samples), dtype=float)   # (N, 2) lat, lon
    return coords[:, 0], coords[:, 1]
