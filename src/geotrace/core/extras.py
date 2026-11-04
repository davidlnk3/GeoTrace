"""Checks for optional dependency groups, so tools can explain how to install what is missing."""

import importlib.util

VERIFICATION_MODULES = ("torch", "torchvision", "safetensors", "numpy", "cv2", "lightglue", "platformdirs")


class MissingExtraError(RuntimeError):
    """Raised when an optional extra is not installed. The message is safe to show the model."""


def require_verification_extra() -> None:
    """Raise MissingExtraError unless every module of the `verification` extra can be imported."""
    missing = [name for name in VERIFICATION_MODULES if importlib.util.find_spec(name) is None]
    if missing:
        raise MissingExtraError(
            "The image verification dependencies are not installed (missing: "
            f"{', '.join(missing)}). Install them in the GeoTrace directory with "
            "`uv sync --extra verification` (or `pip install -e \".[verification]\"`), "
            "then restart the MCP server."
        )
