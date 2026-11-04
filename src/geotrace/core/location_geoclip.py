"""Most probable locations of one image, using GeoCLIP.

GeoCLIP (Vivanco et al., NeurIPS 2023, arXiv:2309.16020) embeds the image with CLIP and
each location of a gallery of 100K GPS coordinates with a location encoder, then scores
every gallery location by image-location similarity. The probabilities are a softmax over
that gallery, so they rank the candidates. They are not calibrated confidences.
"""
import io
from functools import lru_cache

from PIL import Image, ImageOps


@lru_cache(maxsize=1)
def _load():
    from geoclip import GeoCLIP

    # The GeoCLIP weights and gallery ship with the package. The CLIP ViT-L/14 backbone
    # is downloaded from the Hugging Face Hub on first use.
    model = GeoCLIP()
    model.eval()
    return model


def _open_rgb(image_bytes: bytes) -> Image.Image:
    with Image.open(io.BytesIO(image_bytes)) as img:
        return ImageOps.exif_transpose(img).convert("RGB")


def location_guesser_geoclip(image_bytes: bytes, top_k: int = 5) -> dict:
    """Return the top_k most probable (latitude, longitude) for the image, best first.

    GeoCLIP.predict() opens the file without applying its EXIF orientation, so phone photos
    would be scored sideways. This applies the orientation first and then runs the same
    scoring as predict().
    """
    import torch

    model = _load()
    if not 1 <= top_k <= len(model.gps_gallery):
        raise ValueError(f"top_k must be between 1 and {len(model.gps_gallery)}, got {top_k}.")

    image = _open_rgb(image_bytes)
    with torch.inference_mode():
        x = model.image_encoder.preprocess_image(image)
        logits_per_image, _ = model.forward(x, model.gps_gallery)
        probs = logits_per_image.softmax(dim=-1)
        top = torch.topk(probs, top_k, dim=1)

    coords = model.gps_gallery.index_select(0, top.indices[0]).cpu().tolist()
    scores = top.values[0].cpu().tolist()
    candidates = [
        {
            "rank": rank,
            "latitude": round(lat, 6),
            "longitude": round(lon, 6),
            "probability": round(p, 6),
        }
        for rank, ((lat, lon), p) in enumerate(zip(coords, scores), start=1)
    ]
    return {
        "most_probable": candidates[0],
        "candidates": candidates,
        "model": "GeoCLIP",
        "note": "Probabilities are a softmax over a gallery of 100K locations. They rank the "
                "candidates and are not calibrated confidences.",
    }
