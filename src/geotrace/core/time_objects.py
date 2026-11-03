"""Build-time intervals for the objects in one image, using EVA02-CLIP.

The scoring follows the time probing of "A Matter of Time: Revealing the Structure of
Time in Vision-Language Models" (Tekaya, Waldner, Zeppelzauer, arXiv:2510.19559): the
image embedding is compared with text embeddings of "... was built in the year <year>"
prompts, one per candidate year, and the years are scored by similarity.

The paper scores the whole image with prompts that name no object. Here the object name
is put into the prompt, so each object gets its own estimate of its build year. That
extension, and the interval rule below, are not evaluated in the paper.
"""
import io
from datetime import date
from functools import lru_cache

from PIL import Image, ImageOps

MODEL_NAME = "EVA02-L-14-336"   # best time-probing model in the paper (Table 1, MAE 6.20)
PRETRAINED = "merged2b_s6b_b61k"
YEAR_MIN = 1700                 # start of the candidate years in the paper

TEMPLATE = "A photo of a {obj}. Was built in the year {year}."
# TAI tolerance from the paper: a prediction within T(y) years of the truth counts as correct.
# T goes linearly from 20 years at 1700 to 5 years at 2024 (clamped beyond 2024).
TOL_YEARS = (1700, 2024)
TOL_VALUES = (20.0, 5.0)


@lru_cache(maxsize=1)
def _load():
    import open_clip

    model, _, preprocess = open_clip.create_model_and_transforms(MODEL_NAME, pretrained=PRETRAINED)
    model.eval()
    tokenizer = open_clip.get_tokenizer(MODEL_NAME)
    return model, preprocess, tokenizer


def _open_rgb(image_bytes: bytes) -> Image.Image:
    with Image.open(io.BytesIO(image_bytes)) as img:
        return ImageOps.exif_transpose(img).convert("RGB")


def _embed_image(model, preprocess, image_bytes: bytes):
    import torch
    import torch.nn.functional as F

    x = preprocess(_open_rgb(image_bytes)).unsqueeze(0)
    with torch.inference_mode():
        return F.normalize(model.encode_image(x), dim=-1)


def _embed_text(model, tokenizer, texts: list[str]):
    import torch
    import torch.nn.functional as F

    with torch.inference_mode():
        return F.normalize(model.encode_text(tokenizer(texts)), dim=-1)


def _clean_objects(objects: list[str]) -> list[str]:
    names = [o.strip() for o in objects]
    if not names or any(not n for n in names):
        raise ValueError("objects must be a non-empty list of non-empty object names.")
    return names


def time_from_objects(image_bytes: bytes, objects: list[str]) -> dict:
    """Return the build-time interval, up to today, for each object named in `objects`.

    For each object, most_likely_build_year is the year with the highest image-text similarity
    (the paper's rule). built_from_year is that year minus the paper's TAI tolerance, and
    built_until_year is today. `image_taken_not_before_year` is the latest built_from_year: if
    the objects are really in the photo, it cannot be older than the most recent of them.
    """
    import numpy as np

    names = _clean_objects(objects)
    model, preprocess, tokenizer = _load()
    today = date.today().year
    years = list(range(YEAR_MIN, today + 1))

    img = _embed_image(model, preprocess, image_bytes)

    results = []
    for name in names:
        t = _embed_text(model, tokenizer, [TEMPLATE.format(obj=name, year=y) for y in years])
        sims = (img @ t.T).detach().cpu().numpy()[0]
        estimate = years[int(np.argmax(sims))]          # paper: y_pred = argmax_y I^T T_y
        tol = float(np.interp(estimate, TOL_YEARS, TOL_VALUES))
        results.append({
            "object": name,
            "most_likely_build_year": estimate,
            "built_from_year": int(np.floor(estimate - tol)),
            "built_until_year": today,
            "tolerance_years": round(tol, 1),
        })

    return {
        "current_year": today,
        "image_taken_not_before_year": max(r["built_from_year"] for r in results),
        "objects": results,
    }
