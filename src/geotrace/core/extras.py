"""Checks for optional dependency groups, so tools can explain how to install what is missing."""

import importlib.metadata
import importlib.util

REPO_URL = "https://github.com/davidlnk3/GeoTrace"
VERIFICATION_MODULES = ("torch", "torchvision", "safetensors", "numpy", "cv2", "lightglue", "platformdirs")


def install_hint(extra: str) -> str:
    """How to reinstall GeoTrace with an optional extra, pinned to the running version."""
    try:
        ref = f"@v{importlib.metadata.version('geotrace')}"
    except importlib.metadata.PackageNotFoundError:
        ref = ""
    return (
        f"Add the '{extra}' extra to the GeoTrace install: "
        f"`uvx --python 3.12 --from 'geotrace[{extra}] @ git+{REPO_URL}{ref}' geotrace` "
        f"(see {REPO_URL}#install), then restart the MCP server."
    )


class MissingExtraError(RuntimeError):
    """Raised when an optional extra is not installed. The message is safe to show the model."""


def require_verification_extra() -> None:
    """Raise MissingExtraError unless every module of the `verification` extra can be imported."""
    missing = [name for name in VERIFICATION_MODULES if importlib.util.find_spec(name) is None]
    if missing:
        raise MissingExtraError(
            "The image verification dependencies are not installed (missing: "
            f"{', '.join(missing)}). " + install_hint("verification")
        )
