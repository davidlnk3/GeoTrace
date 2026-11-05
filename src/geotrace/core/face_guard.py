"""Face detection used to refuse reverse image searches on people.

This is a safeguard, not a guarantee: Haar cascades miss faces that are small, turned away,
partly hidden, blurred or poorly lit, and the main protection is the tool description telling
the model never to search people. A miss here must not be read as "no person in the crop".

Needs OpenCV from the optional 'reverse-search' extra.
"""
import importlib.util
from functools import cache
from typing import Any

from PIL import Image

from geotrace.core.extras import install_hint

DETECT_MAX_SIDE = 1024   # detection runs on a downscaled copy for speed
INSTALL_HINT = install_hint("reverse-search")


def opencv_available() -> bool:
    return importlib.util.find_spec("cv2") is not None


@cache
def _cascades() -> tuple[Any, ...]:
    import cv2

    names = ("haarcascade_frontalface_default.xml", "haarcascade_profileface.xml")
    return tuple(cv2.CascadeClassifier(cv2.data.haarcascades + n) for n in names)  # type: ignore[attr-defined]


def largest_face_fraction(crop: Image.Image) -> float:
    """Share of the crop's area covered by the largest detected face (0 if none)."""
    import cv2
    import numpy as np

    img = crop.convert("L")
    img.thumbnail((DETECT_MAX_SIDE, DETECT_MAX_SIDE))
    gray = cv2.equalizeHist(np.asarray(img))
    h, w = gray.shape
    min_side = max(20, min(w, h) // 20)
    largest = 0
    for cascade in _cascades():
        for flipped in (False, True):   # the profile cascade only finds faces looking one way
            src = cv2.flip(gray, 1) if flipped else gray
            faces = cascade.detectMultiScale(src, scaleFactor=1.1, minNeighbors=5, minSize=(min_side, min_side))
            for _, _, fw, fh in faces:
                largest = max(largest, int(fw) * int(fh))
    return largest / (w * h)
