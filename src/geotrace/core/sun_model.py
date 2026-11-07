"""Sun altitude and relative azimuth from a single image, using the DINOv2 sun model.

The network and its decoding are a copy of the inference path in the ADOLE
project (allweather/model.py, allweather/predict.py, allweather/geometry.py),
so the checkpoint produced there loads here unchanged.

Torch and transformers are imported lazily, so the rest of GeoTrace still
starts when they are not installed.
"""
import io
import os
from pathlib import Path

# The checkpoint is downloaded from the Hugging Face Hub on first use, pinned to a commit so a new
# upload can't change the results. GEOTRACE_SUN_CHECKPOINT points to a local file instead.
CHECKPOINT_REPO = "davidlnk3/geotrace-sun"
CHECKPOINT_FILE = "best.pt"
CHECKPOINT_REVISION = "d8035af69f5b37aaed290bba5e98bca00d9df0b0"
CHECKPOINT_ENV = "GEOTRACE_SUN_CHECKPOINT"

CROP_SIZE = 128          # input size used in training (allweather/data.py)
BACKBONE_SIZE = 126      # DINOv2 input (multiple of the 14-px patch size)
BACKBONE_ID = "facebook/dinov2-small"
N_AZ = 64
N_EL = 16
EL_MAX = 90.0
AZ_BIN = 360.0 / N_AZ
EL_BIN = EL_MAX / N_EL

_model = None


def _bin_centers():
    import numpy as np

    az = (np.arange(N_AZ) + 0.5) * AZ_BIN
    el = (np.arange(N_EL) + 0.5) * EL_BIN
    return az, el


def _build_network():
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from transformers import AutoConfig, AutoModel

    class SunNetDINOv2(nn.Module):
        def __init__(self):
            super().__init__()
            # Random weights with the dinov2-small architecture; the checkpoint supplies all values.
            self.backbone = AutoModel.from_config(AutoConfig.from_pretrained(BACKBONE_ID))
            self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
            self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
            d = self.backbone.config.hidden_size
            self.fc = nn.Sequential(nn.Linear(2 * d, 512), nn.ELU(inplace=True),
                                    nn.Linear(512, 512), nn.ELU(inplace=True))
            self.head_az = nn.Linear(512, N_AZ)
            self.head_el = nn.Linear(512, N_EL)
            az_c, el_c = _bin_centers()
            self.register_buffer("az_centers", torch.tensor(az_c, dtype=torch.float32))
            self.register_buffer("el_centers", torch.tensor(el_c, dtype=torch.float32))

        def forward(self, x):
            x = F.interpolate(x, size=(BACKBONE_SIZE, BACKBONE_SIZE), mode="bilinear",
                              align_corners=False)
            x = (x - self.mean) / self.std
            tok = self.backbone(pixel_values=x).last_hidden_state
            feat = torch.cat([tok[:, 0], tok[:, 1:].mean(dim=1)], dim=1)
            h = self.fc(feat)
            return self.head_az(h), self.head_el(h)

        @torch.no_grad()
        def decode(self, az_logits, el_logits):
            # Altitude: probability-weighted mean of elevation bin centres.
            # Azimuth: circular mean of azimuth bin centres (359 and 1 average to 0).
            p_az = F.softmax(az_logits, dim=1)
            p_el = F.softmax(el_logits, dim=1)
            alt = (p_el * self.el_centers).sum(1)
            rad = torch.deg2rad(self.az_centers)
            x = (p_az * torch.cos(rad)).sum(1)
            y = (p_az * torch.sin(rad)).sum(1)
            az = torch.remainder(torch.rad2deg(torch.atan2(y, x)), 360.0)
            return alt, az

    return SunNetDINOv2()


def checkpoint_path() -> Path:
    """Return the local checkpoint path, downloading it from the Hub when no local file is set."""
    local = os.environ.get(CHECKPOINT_ENV)
    if local:
        path = Path(local).expanduser()
        if not path.is_file():
            raise FileNotFoundError(f"Sun model checkpoint not found at {path} (set by {CHECKPOINT_ENV}).")
        return path

    from huggingface_hub import hf_hub_download

    return Path(hf_hub_download(CHECKPOINT_REPO, CHECKPOINT_FILE, revision=CHECKPOINT_REVISION))


def load_model():
    """Load the checkpoint once and keep it in memory."""
    global _model
    if _model is not None:
        return _model

    import torch

    state = torch.load(checkpoint_path(), map_location="cpu", weights_only=True)
    if state.get("arch") != "dinov2":
        raise ValueError(f"Checkpoint arch is {state.get('arch')!r}; expected 'dinov2'.")

    model = _build_network()
    model.load_state_dict(state["model"])
    model.eval()
    _model = model
    return model


def _to_square_rgb(image_bytes: bytes):
    import numpy as np
    from PIL import Image, ImageOps
    try:
        import pillow_heif  # noqa: F401  (registers HEIF support; core/images.py also does this)
    except ImportError:
        pass

    with Image.open(io.BytesIO(image_bytes)) as img:
        img = ImageOps.exif_transpose(img).convert("RGB")
        w, h = img.size
        s = min(w, h)
        left, top = (w - s) // 2, (h - s) // 2
        img = img.crop((left, top, left + s, top + s))
        img = img.resize((CROP_SIZE, CROP_SIZE), Image.BILINEAR)
        return np.asarray(img)


def estimate(image_bytes: bytes) -> dict:
    """Return the sun altitude above the horizon and the sun azimuth relative to the
    camera's optical axis (clockwise, [0, 360)) for a photo of an outdoor scene.

    The centre square of the image is used, as in training. The model estimates the
    sun from scene cues only; the sun is assumed not to be visible in the frame.
    """
    import torch

    model = load_model()
    rgb = _to_square_rgb(image_bytes)
    x = torch.from_numpy(rgb.copy()).permute(2, 0, 1).float().unsqueeze(0) / 255.0
    az_logits, el_logits = model(x)
    alt, az = model.decode(az_logits, el_logits)
    return {
        "sun_altitude_deg": round(float(alt[0]), 2),
        "sun_relative_azimuth_to_camera_deg": round(float(az[0]), 2),
    }
