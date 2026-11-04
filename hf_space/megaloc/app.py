"""MegaLoc global descriptors on ZeroGPU, for GeoTrace's verify_candidate_locations.

MegaLoc: Berton & Masone, "MegaLoc: One Retrieval to Place Them All", CVPRW 2025.
Code and weights: https://github.com/gmberton/MegaLoc, https://huggingface.co/gberton/MegaLoc (MIT).

The Space only computes descriptors; ranking and caching happen in GeoTrace. The model code is
pinned to the same commit as GeoTrace's local backend, so both give the same descriptors and
share one embedding cache.
"""

import base64

import gradio as gr
import spaces  # must be imported before torch
import torch
import torch.nn.functional as F
import torchvision.transforms as tfm
from PIL import Image, ImageOps

MEGALOC_COMMIT = "5fe0dd697c4a70ba3e23607f6716ab3c606b16db"
MODEL_TAG = f"megaloc-{MEGALOC_COMMIT[:12]}"
INPUT_SIZE = 322
BATCH_SIZE = 64
MAX_IMAGES = 256
MAX_PIXELS = 4096 * 4096

Image.MAX_IMAGE_PIXELS = MAX_PIXELS

model = torch.hub.load(
    f"gmberton/MegaLoc:{MEGALOC_COMMIT}", "get_trained_model", trust_repo=True, skip_validation=True
)
model = model.eval().to("cuda")

# Evaluation preprocessing from the MegaLoc README.
transform = tfm.Compose([
    tfm.ToTensor(),
    tfm.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    tfm.Resize(size=[INPUT_SIZE, INPUT_SIZE], antialias=True),
])


def _load(path: str) -> torch.Tensor | None:
    try:
        with Image.open(path) as img:
            return transform(ImageOps.exif_transpose(img).convert("RGB"))
    except Exception:  # noqa: BLE001 - an unreadable file gets a null descriptor
        return None


def _duration(images, *args, **kwargs) -> int:
    return min(120, 10 + len(images or []) // 8)


@spaces.GPU(duration=_duration)
def embed(images: list[str] | None) -> dict:
    """Compute L2-normalised MegaLoc descriptors for up to 256 images.

    Args:
        images: image files (JPEG, PNG or WebP).

    Returns:
        {"model": tag, "dim": 8448, "dtype": "float16", "vectors": [...]}, where each vector is
        the base64 of little-endian float16 values, in the order of the inputs, or null for an
        image that could not be decoded.
    """
    paths = list(images or [])
    if len(paths) > MAX_IMAGES:
        raise gr.Error(f"At most {MAX_IMAGES} images per call.")
    tensors = [_load(p) for p in paths]
    valid = [i for i, t in enumerate(tensors) if t is not None]
    vectors: list[str | None] = [None] * len(paths)
    dim = 0
    if valid:
        batch = torch.stack([tensors[i] for i in valid])
        out = []
        with torch.inference_mode():
            for start in range(0, len(batch), BATCH_SIZE):
                chunk = batch[start:start + BATCH_SIZE].to("cuda")
                out.append(F.normalize(model(chunk).float(), dim=-1).cpu())
        desc = torch.cat(out).numpy().astype("<f2")
        dim = int(desc.shape[1])
        for i, vec in zip(valid, desc):
            vectors[i] = base64.b64encode(vec.tobytes()).decode("ascii")
    return {"model": MODEL_TAG, "dim": dim, "dtype": "float16", "vectors": vectors}


demo = gr.Interface(
    fn=embed,
    inputs=gr.File(file_count="multiple", file_types=["image"], type="filepath", label="Images"),
    outputs=gr.JSON(label="Descriptors"),
    title="MegaLoc descriptors",
    description=(
        "Global place-recognition descriptors from [MegaLoc](https://github.com/gmberton/MegaLoc) "
        "(Berton & Masone, CVPRW 2025, MIT). Used by GeoTrace to rank street-level reference photos. "
        "Returns float16 L2-normalised vectors (8448-d), base64-encoded."
    ),
    api_name="embed",
    flagging_mode="never",
)

if __name__ == "__main__":
    demo.queue(max_size=32).launch()
