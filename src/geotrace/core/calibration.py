import functools
import io
import math
from dataclasses import dataclass

from PIL import Image


@dataclass
class CameraCalibration:
    roll_deg: float                    # rotation around the optical axis; 0 = level image
    pitch_deg: float                   # positive = camera looking up, negative = looking down
    vertical_fov_deg: float
    pitch_uncertainty_deg: float       # standard deviation of the pitch estimate
    vertical_fov_uncertainty_deg: float  # standard deviation of the vertical FOV estimate


@functools.cache
def _model():
    # Imported here so the server runs without torch / geocalib installed
    import torch
    from geocalib import GeoCalib

    device = "cuda" if torch.cuda.is_available() else "cpu"
    return GeoCalib(weights="pinhole").to(device), device


def _load_tensor(image: bytes, device: str):
    import numpy as np
    import torch

    with Image.open(io.BytesIO(image)) as img:
        rgb = img.convert("RGB")
    arr = np.asarray(rgb, dtype=np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1).to(device)


def calibrate(image: bytes):
    """Estimate the camera with GeoCalib.

    Returns the calibration, the 3x3 intrinsic matrix K and the unit vector pointing
    down in camera coordinates (x right, y down, z forward). GeoCalib's gravity is the
    up vector, so it is negated here. The first run downloads the GeoCalib weights.
    """
    model, device = _model()
    out = model.calibrate(_load_tensor(image, device))

    camera, gravity = out["camera"], out["gravity"]
    calibration = CameraCalibration(
        roll_deg=math.degrees(gravity.roll[0].item()),
        pitch_deg=math.degrees(gravity.pitch[0].item()),
        vertical_fov_deg=math.degrees(camera.vfov[0].item()),
        pitch_uncertainty_deg=math.degrees(out["pitch_uncertainty"][0].item()),
        vertical_fov_uncertainty_deg=math.degrees(out["vfov_uncertainty"][0].item()),
    )
    K = camera.K[0].cpu().numpy()
    down = -gravity.vec3d[0].cpu().numpy()
    return calibration, K, down


def raw_output(image: bytes) -> dict:
    """GeoCalib's raw estimates for an image, for checking them by hand."""
    model, device = _model()
    out = model.calibrate(_load_tensor(image, device))

    camera, gravity, conf = out["camera"], out["gravity"], out["up_confidence"]
    deg = lambda t: math.degrees(t[0].item())
    return {
        "size_px": camera.size[0].tolist(),
        "focal_px": camera.f[0].tolist(),
        "principal_point_px": camera.c[0].tolist(),
        "K": camera.K[0].tolist(),
        "vfov_deg": deg(camera.vfov),
        "hfov_deg": deg(camera.hfov),
        "gravity_vector": gravity.vec3d[0].tolist(),
        "roll_deg": deg(gravity.roll),
        "pitch_deg": deg(gravity.pitch),
        "uncertainty": {
            "roll_deg": deg(out["roll_uncertainty"]),
            "pitch_deg": deg(out["pitch_uncertainty"]),
            "gravity_deg": deg(out["gravity_uncertainty"]),
            "vfov_deg": deg(out["vfov_uncertainty"]),
            "focal_px": out["focal_uncertainty"][0].item(),
        },
        "up_confidence": {
            "min": conf.min().item(),
            "mean": conf.mean().item(),
            "max": conf.max().item(),
        },
    }
