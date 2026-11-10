"""Capture-year estimate for one photograph, from the look of the photograph itself.

This follows "Blind Dates: Examining the Expression of Temporality in Historical Photographs"
(Barancova, Wevers, van Noord, CHR 2023, arXiv:2310.06633): a multinomial logistic regression
over one year class per year, on top of frozen OpenCLIP image embeddings. The paper reports a
mean absolute error of 6.65 years for this fine-tuned head, against 15.8 for zero-shot prompting.

The head is trained here on the De Boer Scene Detection press-photograph corpus plus a sample of
Date Estimation in the Wild images, and is published as a weight matrix and a bias vector.

IMPORTANT LIMIT: the head can only ever answer with a year inside its trained range (1950-1999),
because that range is its entire label space. It cannot represent a photograph from 2010, and on
one it still returns some year in range. The caller must treat the answer as "if this photograph
is from the analogue era, this is when it looks like it was taken" and weigh `confidence` and
`out_of_range_warning` accordingly. It reads photographic style, not the content of the scene, so
it is evidence about the medium rather than proof about the date.
"""
import io
from functools import lru_cache

from PIL import Image, ImageOps

REPO_ID = "davidlnk3/geotrace-photo-year"
# Pinned so an update to the model repo cannot silently change what this tool reports.
REVISION = "b834c1001f888c945ffefe3c52918266ceebdbf9"
WEIGHTS_FILE = "model.safetensors"
CONFIG_FILE = "config.json"

MODEL_NAME = "ViT-B-32-quickgelu"   # the paper's backbone (CANAL-amsterdam/dating-images)
PRETRAINED = "laion400m_e32"
L2_NORMALIZE = False                # the reference implementation feeds raw encode_image output

INTERVAL_MASS = 0.8                 # central probability mass reported as a year interval


@lru_cache(maxsize=1)
def _load():
    """Return (clip_model, preprocess, weight, bias, years, config). Cached for the process."""
    import json

    import open_clip
    from huggingface_hub import hf_hub_download
    from safetensors.numpy import load_file

    model, _, preprocess = open_clip.create_model_and_transforms(MODEL_NAME, pretrained=PRETRAINED)
    model.eval()

    tensors = load_file(hf_hub_download(repo_id=REPO_ID, filename=WEIGHTS_FILE, revision=REVISION))
    with open(hf_hub_download(repo_id=REPO_ID, filename=CONFIG_FILE, revision=REVISION)) as fh:
        config = json.load(fh)
    return model, preprocess, tensors["weight"], tensors["bias"], tensors["classes"], config


def _open_rgb(image_bytes: bytes) -> Image.Image:
    with Image.open(io.BytesIO(image_bytes)) as img:
        return ImageOps.exif_transpose(img).convert("RGB")


def _interval(years, probs, mass: float) -> list[int]:
    """Smallest contiguous run of years holding at least `mass` of the probability."""
    best = (len(years), int(years[0]), int(years[-1]))
    lo = 0
    total = 0.0
    for hi in range(len(years)):
        total += probs[hi]
        while total - probs[lo] >= mass:
            total -= probs[lo]
            lo += 1
        if total >= mass and (hi - lo) < best[0]:
            best = (hi - lo, int(years[lo]), int(years[hi]))
    return [best[1], best[2]]


def predict_from_embedding(emb, weight, bias, years, mae=None) -> dict:
    """Run the linear year head over one image embedding. Pure numpy, so it is testable alone."""
    import numpy as np

    logits = emb @ weight.T + bias
    probs = np.exp(logits - logits.max())
    probs /= probs.sum()

    top = int(np.argmax(probs))
    year_min, year_max = int(years[0]), int(years[-1])

    return {
        "estimated_year": int(years[top]),
        "expected_year": round(float(probs @ years), 1),
        "interval_80pct": _interval(years, probs, INTERVAL_MASS),
        "confidence": round(float(probs[top]), 4),
        "model_year_range": [year_min, year_max],
        "model_mae_years": mae,
        "method": "photographic style (OpenCLIP embedding + logistic year classifier)",
        "out_of_range_warning": (
            f"This model only has year classes {year_min}-{year_max}, so it always answers inside "
            f"that range even for a photograph taken outside it. Treat the estimate as how old the "
            f"photograph looks, not as proof of its date, and ignore it if other evidence puts the "
            f"photo after {year_max}."
        ),
    }


def estimate_capture_year(image_bytes: bytes) -> dict:
    """Estimate the year the photograph was taken from its visual style.

    Returns the argmax year (the paper's rule), the probability-weighted expected year, a central
    interval, the confidence of the argmax year, and the range the head is able to answer within.
    """
    import torch

    model, preprocess, weight, bias, years, config = _load()

    x = preprocess(_open_rgb(image_bytes)).unsqueeze(0)
    with torch.inference_mode():
        feats = model.encode_image(x)
    if L2_NORMALIZE:
        feats = torch.nn.functional.normalize(feats, dim=-1)
    emb = feats.float().cpu().numpy()[0]

    mae = config.get("metrics", {}).get("final", {}).get("mae_years")
    return predict_from_embedding(emb, weight, bias, years, mae)
