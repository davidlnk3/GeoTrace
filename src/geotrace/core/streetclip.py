"""Most probable location of one image, zero-shot, using StreetCLIP.

StreetCLIP (Haas et al., 2023, https://huggingface.co/geolocal/StreetCLIP) is a CLIP model
further pretrained on street-view images for geolocalization. It is used without training
here: the image is compared with one text prompt per candidate place, and the similarities
are turned into probabilities with a softmax over the candidates. The probabilities are
relative to the candidate set, not calibrated over the whole world.

The model is downloaded from Hugging Face on first use. Torch and transformers are imported
lazily, so the rest of GeoTrace still starts when they are not installed.
"""
import io
from functools import lru_cache

from PIL import Image, ImageOps

from geotrace.core.countries import COUNTRIES

MODEL_ID = "geolocal/StreetCLIP"
TEMPLATE = "a street view photo taken in {}."


@lru_cache(maxsize=1)
def _load():
    from transformers import CLIPImageProcessor, CLIPModel, CLIPProcessor, CLIPTokenizerFast

    model = CLIPModel.from_pretrained(MODEL_ID)
    model.eval()
    # Built from its parts: CLIPProcessor.from_pretrained picks a tokenizer class that
    # fails to load with the transformers 5.x release installed here.
    processor = CLIPProcessor(
        image_processor=CLIPImageProcessor.from_pretrained(MODEL_ID),
        tokenizer=CLIPTokenizerFast.from_pretrained(MODEL_ID),
    )
    return model, processor


def _open_rgb(image_bytes: bytes) -> Image.Image:
    with Image.open(io.BytesIO(image_bytes)) as img:
        return ImageOps.exif_transpose(img).convert("RGB")


def _candidates(custom: list[str] | None) -> list[tuple[str, float | None, float | None]]:
    """Return (name, latitude, longitude) for each candidate. Custom names have no coordinates."""
    if custom is None:
        return [(name, lat, lon) for name, lat, lon in COUNTRIES]
    names = []
    for raw in custom:
        name = raw.strip()
        if not name:
            raise ValueError("candidates must not contain empty names.")
        if name not in names:
            names.append(name)
    if len(names) < 2:
        raise ValueError("candidates must contain at least two distinct names.")
    return [(name, None, None) for name in names]


def locate(image_bytes: bytes, candidates: list[str] | None = None, top_k: int = 5) -> dict:
    """Score the image against each candidate place and return the most probable ones.

    With no `candidates`, the built-in list of countries is used, and each result carries the
    coordinates of a representative point for that country. With custom `candidates`, such as
    ["Lisbon, Portugal", "Porto, Portugal"], the names are scored as given and have no coordinates.
    """
    import torch

    places = _candidates(candidates)
    if not 1 <= top_k <= len(places):
        raise ValueError(f"top_k must be between 1 and the number of candidates ({len(places)}).")

    model, processor = _load()
    image = _open_rgb(image_bytes)
    texts = [TEMPLATE.format(name) for name, _, _ in places]
    inputs = processor(text=texts, images=image, return_tensors="pt", padding=True)
    with torch.inference_mode():
        logits = model(**inputs).logits_per_image[0]
    probs = logits.softmax(dim=0).tolist()

    ranked = sorted(zip(places, probs), key=lambda item: item[1], reverse=True)[:top_k]
    results = [
        {
            "location": name,
            "latitude": lat,
            "longitude": lon,
            "probability": round(p, 4),
        }
        for (name, lat, lon), p in ranked
    ]

    return {
        "model": MODEL_ID,
        "candidate_set": "built-in countries" if candidates is None else "custom",
        "num_candidates": len(places),
        "most_probable": results[0],
        "top": results,
    }
