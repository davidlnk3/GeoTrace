"""MegaLoc global descriptors on ZeroGPU, for GeoTrace's verify_candidate_locations.

MegaLoc: Berton & Masone, "MegaLoc: One Retrieval to Place Them All", CVPRW 2025.
Code and weights: https://github.com/gmberton/MegaLoc, https://huggingface.co/gberton/MegaLoc (MIT).

The Space only computes descriptors; ranking and caching happen in GeoTrace. The model code is
pinned to the same commit as GeoTrace's local backend, so both give the same descriptors and
share one embedding cache.
"""

import base64
import io
import zipfile

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


MAX_MEMBER_BYTES = 10 * 1024 * 1024
MAX_ARCHIVE_BYTES = 256 * 1024 * 1024


def _load(source) -> torch.Tensor | None:
    """A path or a file object -> preprocessed tensor, or None if it is not a readable image."""
    try:
        with Image.open(source) as img:
            return transform(ImageOps.exif_transpose(img).convert("RGB"))
    except Exception:  # noqa: BLE001 - an unreadable file gets a null descriptor
        return None


def _describe(tensors: list[torch.Tensor | None]) -> dict:
    """Run MegaLoc on the decodable tensors; the JSON answer shared by both endpoints."""
    valid = [i for i, t in enumerate(tensors) if t is not None]
    vectors: list[str | None] = [None] * len(tensors)
    dim = 0
    if valid:
        batch = torch.stack([t for t in tensors if t is not None])
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


def _zip_members(archive: str) -> list[zipfile.ZipInfo]:
    """The archive's files in name order, after the size checks (against zip bombs)."""
    with zipfile.ZipFile(archive) as zf:
        members = sorted((m for m in zf.infolist() if not m.is_dir()), key=lambda m: m.filename)
    if len(members) > MAX_IMAGES:
        raise gr.Error(f"At most {MAX_IMAGES} images per call.")
    if any(m.file_size > MAX_MEMBER_BYTES for m in members):
        raise gr.Error(f"Each image must be at most {MAX_MEMBER_BYTES // (1024 * 1024)} MB.")
    if sum(m.file_size for m in members) > MAX_ARCHIVE_BYTES:
        raise gr.Error("The archive is too large.")
    return members


def _duration_files(images, *args, **kwargs) -> int:
    return min(120, 10 + len(images or []) // 8)


def _duration_zip(archive, *args, **kwargs) -> int:
    try:
        with zipfile.ZipFile(archive) as zf:
            n = len(zf.infolist())
    except Exception:  # noqa: BLE001 - the endpoint itself reports a bad archive
        n = 0
    return min(120, 10 + min(n, MAX_IMAGES) // 8)


@spaces.GPU(duration=_duration_files)
def embed(images: list[str] | None) -> dict:
    """Compute L2-normalised MegaLoc descriptors for up to 256 image files.

    Each file is a separate upload, which is slow for many images: use /embed_zip for batches.

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
    return _describe([_load(p) for p in paths])


@spaces.GPU(duration=_duration_zip)
def embed_zip(archive: str | None) -> dict:
    """Compute L2-normalised MegaLoc descriptors for up to 256 images sent as one ZIP archive.

    Args:
        archive: a ZIP of image files (JPEG, PNG or WebP). They are processed in file-name order,
            so name them 0000.jpg, 0001.jpg, ... to keep their order.

    Returns:
        The same answer as /embed, with one vector (or null) per file of the archive, in name order.
    """
    if not archive:
        return _describe([])
    try:
        members = _zip_members(archive)
    except zipfile.BadZipFile as exc:
        raise gr.Error("Not a valid ZIP archive.") from exc
    with zipfile.ZipFile(archive) as zf:
        tensors = [_load(io.BytesIO(zf.read(m))) for m in members]
    return _describe(tensors)


DESCRIPTION = (
    "Global place-recognition descriptors from [MegaLoc](https://github.com/gmberton/MegaLoc) "
    "(Berton & Masone, CVPRW 2025, MIT). Used by GeoTrace to rank street-level reference photos. "
    "Returns float16 L2-normalised vectors (8448-d), base64-encoded."
)

files_tab = gr.Interface(
    fn=embed,
    inputs=gr.File(file_count="multiple", file_types=["image"], type="filepath", label="Images"),
    outputs=gr.JSON(label="Descriptors"),
    description=DESCRIPTION,
    api_name="embed",
    flagging_mode="never",
)
zip_tab = gr.Interface(
    fn=embed_zip,
    inputs=gr.File(file_types=[".zip"], type="filepath", label="ZIP of images"),
    outputs=gr.JSON(label="Descriptors"),
    description=DESCRIPTION + " One upload for the whole batch: the fast path for API clients.",
    api_name="embed_zip",
    flagging_mode="never",
)
demo = gr.TabbedInterface([files_tab, zip_tab], ["Images", "ZIP archive"], title="MegaLoc descriptors")

if __name__ == "__main__":
    demo.queue(max_size=32).launch()
