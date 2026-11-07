"""The DINOv2 sun model tool: the missing-extra message and the result keys."""

import io

import pytest
from mcp.server.mcpserver.exceptions import ToolError
from PIL import Image

from geotrace.core import extras, sun_model
from geotrace.tools.relative_sun_position_from_outdoor_lighting import (
    relative_sun_position_from_outdoor_lighting,
)

# The keys estimate() returns. The tool reads both, so a rename on one side only is a KeyError.
RESULT_KEYS = {"sun_altitude_deg", "sun_relative_azimuth_to_camera_deg"}


class FakeContext:
    def __init__(self):
        self.messages = []

    async def info(self, message):
        self.messages.append(message)


def png_bytes():
    buf = io.BytesIO()
    Image.new("RGB", (64, 48), "white").save(buf, format="PNG")
    return buf.getvalue()


def test_require_sun_extra_names_the_missing_module(monkeypatch):
    monkeypatch.setattr(extras.importlib.util, "find_spec", lambda name: None)
    with pytest.raises(extras.MissingExtraError, match="transformers"):
        extras.require_sun_extra()


@pytest.mark.anyio
async def test_missing_extra_raises_tool_error(monkeypatch):
    def missing():
        raise extras.MissingExtraError("install with uv sync --extra sun")

    monkeypatch.setattr(extras, "require_sun_extra", missing)
    with pytest.raises(ToolError, match="uv sync --extra sun"):
        await relative_sun_position_from_outdoor_lighting("photo.jpg", FakeContext())  # type: ignore[arg-type]


@pytest.mark.anyio
async def test_returns_both_result_keys(monkeypatch):
    """The tool logs both keys, so estimate() must keep producing exactly them."""
    torch = pytest.importorskip("torch")   # torch ships with several extras; transformers is not needed

    class StubModel:
        """Stands in for the checkpoint: estimate() only calls it and decode()."""

        def __call__(self, x):
            return torch.zeros(1, sun_model.N_AZ), torch.zeros(1, sun_model.N_EL)

        def decode(self, az_logits, el_logits):
            return torch.tensor([33.5]), torch.tensor([217.25])

    monkeypatch.setattr(sun_model, "load_model", StubModel)
    monkeypatch.setattr(
        "geotrace.tools.relative_sun_position_from_outdoor_lighting.load_image",
        _async_return(png_bytes()),
    )

    ctx = FakeContext()
    result = await relative_sun_position_from_outdoor_lighting("photo.jpg", ctx)  # type: ignore[arg-type]
    assert set(result) == RESULT_KEYS
    assert result == {"sun_altitude_deg": 33.5, "sun_relative_azimuth_to_camera_deg": 217.25}
    assert "33.5" in ctx.messages[-1] and "217.25" in ctx.messages[-1]


def _async_return(value):
    async def loader(_source):
        return value

    return loader
