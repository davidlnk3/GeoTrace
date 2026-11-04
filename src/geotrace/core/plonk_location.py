"""Most probable geolocation of an image, using PLONK.

PLONK ("Around the World in 80 Timesteps", arXiv:2412.06781, MIT licence, code at
github.com/nicolas-dufour/plonk, pip package diff-plonk) generates a coordinate on the sphere
by denoising random noise, conditioned on the image. One sample is one plausible location, so
the most probable one is chosen as in the paper's ensemble sampling: draw several samples and
keep the one with the highest likelihood under the model.
"""
import contextlib
import io
import sys
import threading
from functools import lru_cache

from PIL import Image, ImageOps

MODEL_ID = "nicolas-dufour/PLONK_YFCC"   # trained on YFCC-100M, uses a DINOv2-L encoder
DEFAULT_CANDIDATES = 16                  # the paper uses 32; 16 keeps CPU runs to a few minutes

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


def guess_location(image_bytes: bytes, num_candidates: int = DEFAULT_CANDIDATES) -> dict:
    """Return the most probable latitude and longitude of the image.

    Draws `num_candidates` samples from the model and returns the one with the highest
    log-likelihood. log_likelihood is the model's normalised score for that coordinate, so
    higher is more probable; it is only comparable between candidates for the same image.
    """
    if num_candidates < 1:
        raise ValueError("num_candidates must be at least 1.")

    import numpy as np

    pipe = _load_pipeline()
    img = _open_rgb(image_bytes)

    # PLONK prints a progress line to stdout, which would corrupt the MCP stdio protocol.
    with contextlib.redirect_stdout(sys.stderr):
        coords = np.asarray(pipe(img, batch_size=num_candidates), dtype=float)   # (N, 2) lat, lon
        logp = pipe.compute_likelihood(img, coords).detach().cpu().numpy()

    best = int(np.argmax(logp))
    return {
        "latitude": round(float(coords[best, 0]), 5),
        "longitude": round(float(coords[best, 1]), 5),
        "log_likelihood": round(float(logp[best]), 4),
        "num_candidates": num_candidates,
        "model": MODEL_ID,
    }
