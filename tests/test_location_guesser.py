"""The merged location_guesser must send each kind to the PLONK model trained on that imagery.

A swapped mapping would still return plausible regions, so it is checked explicitly.
"""
import pytest

pytest.importorskip("numpy")

from geotrace.core.plonk_location import DEFAULT_SAMPLES, INAT_MODEL, OSV_MODEL, YFCC_MODEL
from geotrace.tools import location_guesser as lg


class FakeContext:
    def __init__(self):
        self.messages = []

    async def info(self, message):
        self.messages.append(message)


@pytest.fixture
def calls(monkeypatch):
    seen = []

    async def fake_load(image):
        return b"jpeg"

    def fake_describe(data, num_samples, model_id):
        seen.append((num_samples, model_id))
        return {"regions": []}

    monkeypatch.setattr(lg, "load_image", fake_load)
    monkeypatch.setattr(lg, "describe_location", fake_describe)
    return seen


@pytest.mark.parametrize(
    "kind,model_id",
    [("street", OSV_MODEL), ("general", YFCC_MODEL), ("nature", INAT_MODEL)],
)
async def test_kind_selects_its_model(calls, kind, model_id):
    await lg.location_guesser(image="photo.jpg", kind=kind, ctx=FakeContext())  # type: ignore[arg-type]
    assert calls == [(DEFAULT_SAMPLES, model_id)]


async def test_num_samples_is_passed_through(calls):
    await lg.location_guesser(image="photo.jpg", kind="nature", ctx=FakeContext(), num_samples=250)  # type: ignore[arg-type]
    assert calls == [(250, INAT_MODEL)]


def test_every_kind_is_mapped():
    assert set(lg.KINDS) == {"street", "general", "nature"}
