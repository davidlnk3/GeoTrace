"""Tests for the Space retrieval backend, with a fake gradio client: no network, no model."""

import base64
import io
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from geotrace.core import mapillary, retrieval
from geotrace.core.mapillary import MapillaryImage
from geotrace.core.retrieval import EmbeddingCache, SpaceMegaLoc

DIM = 8


def jpeg(color: tuple[int, int, int]) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (32, 24), color).save(buf, format="JPEG")
    return buf.getvalue()


def ref(image_id: str) -> MapillaryImage:
    return MapillaryImage(image_id, 0.0, 0.0, None, None, None, False, f"https://t/{image_id}", None)


def unit(i: int) -> np.ndarray:
    v = np.zeros(DIM, dtype=np.float32)
    v[i] = 1.0
    return v


class FakeClient:
    """Answers like the Space: one float16 base64 vector per uploaded file, chosen by its colour."""

    def __init__(self, model_tag: str = retrieval.MODEL_TAG):
        self.calls: list[int] = []
        self.model_tag = model_tag

    def predict(self, images, api_name):
        assert api_name == "/embed"
        self.calls.append(len(images))
        vectors = []
        for f in images:
            path = f["path"] if isinstance(f, dict) else f
            with Image.open(path) as im:
                pixel = im.convert("RGB").getpixel((5, 5))
            assert isinstance(pixel, tuple)
            r, g = pixel[0], pixel[1]
            # Red-ish images point along axis 0, green-ish along axis 1, the rest along axis 2.
            axis = 0 if r > 128 and g < 128 else 1 if g > 128 and r < 128 else 2
            vectors.append(base64.b64encode(unit(axis).astype("<f2").tobytes()).decode())
        return {"model": self.model_tag, "dim": DIM, "dtype": "float16", "vectors": vectors}


@pytest.fixture
def thumbs(monkeypatch):
    """Serve thumbnails from a dict instead of Mapillary; records which ids were requested."""
    store: dict[str, bytes] = {}
    requested: list[str] = []

    async def fake_download(images, size, client=None):
        assert size == 256
        requested.extend(i.id for i in images)
        return {i.id: store[i.id] for i in images if i.id in store}

    monkeypatch.setattr(mapillary, "download_thumbnails", fake_download)
    return store, requested


@pytest.mark.anyio
async def test_space_backend_ranks_and_caches(tmp_path: Path, thumbs):
    store, requested = thumbs
    store.update({"red": jpeg((220, 10, 10)), "green": jpeg((10, 220, 10)), "blue": jpeg((10, 10, 220))})
    client = FakeClient()
    backend = SpaceMegaLoc(
        "someone/space", cache=EmbeddingCache(tmp_path / "e.sqlite3"), client_factory=lambda _: client,
    )
    refs = [ref("green"), ref("red"), ref("blue")]

    ranked = await backend.rank(jpeg((230, 20, 20)), refs, top_k=2)
    assert [r.image.id for r in ranked] == ["red", "green"]
    assert ranked[0].score == pytest.approx(1.0)
    assert client.calls == [4]                       # query + 3 thumbnails

    # Second time: descriptors come from the cache, only the query is uploaded.
    requested.clear()
    ranked = await backend.rank(jpeg((20, 230, 20)), refs, top_k=1)
    assert [r.image.id for r in ranked] == ["green"]
    assert requested == []
    assert client.calls == [4, 1]


@pytest.mark.anyio
async def test_space_backend_rejects_another_model(tmp_path: Path, thumbs):
    store, _ = thumbs
    store["a"] = jpeg((200, 0, 0))
    backend = SpaceMegaLoc(
        "someone/space", cache=EmbeddingCache(tmp_path / "e.sqlite3"),
        client_factory=lambda _: FakeClient(model_tag="megaloc-other"),
    )
    with pytest.raises(retrieval.RetrievalError, match="same MegaLoc commit"):
        await backend.rank(jpeg((200, 0, 0)), [ref("a")], top_k=1)


@pytest.mark.anyio
async def test_space_backend_chunks_large_uploads(tmp_path: Path, thumbs, monkeypatch):
    store, _ = thumbs
    monkeypatch.setattr(retrieval, "SPACE_MAX_IMAGES", 2)
    for i in range(4):
        store[f"r{i}"] = jpeg((200, 0, 0))
    client = FakeClient()
    backend = SpaceMegaLoc(
        "someone/space", cache=EmbeddingCache(tmp_path / "e.sqlite3"), client_factory=lambda _: client,
    )
    ranked = await backend.rank(jpeg((200, 0, 0)), [ref(f"r{i}") for i in range(4)], top_k=10)
    assert len(ranked) == 4
    assert client.calls == [2, 2, 1]                  # 5 images, at most 2 per call


@pytest.mark.anyio
async def test_space_backend_failure_is_actionable(tmp_path: Path, thumbs):
    store, _ = thumbs
    store["a"] = jpeg((200, 0, 0))

    class Down:
        def predict(self, images, api_name):
            raise ConnectionError("Space is sleeping")

    backend = SpaceMegaLoc(
        "someone/space", cache=EmbeddingCache(tmp_path / "e.sqlite3"), client_factory=lambda _: Down(),
    )
    with pytest.raises(retrieval.RetrievalError, match="GEOTRACE_RETRIEVAL_BACKEND=local"):
        await backend.rank(jpeg((200, 0, 0)), [ref("a")], top_k=1)


def test_backend_selection(monkeypatch):
    retrieval.set_backend(None)
    try:
        monkeypatch.setenv("GEOTRACE_RETRIEVAL_BACKEND", "space")
        monkeypatch.setenv("GEOTRACE_MEGALOC_SPACE", "me/my-megaloc")
        backend = retrieval.get_backend()
        assert isinstance(backend, SpaceMegaLoc) and backend.space_id == "me/my-megaloc"

        retrieval.set_backend(None)
        monkeypatch.delenv("GEOTRACE_RETRIEVAL_BACKEND")
        assert isinstance(retrieval.get_backend(), retrieval.LocalMegaLoc)

        retrieval.set_backend(None)
        monkeypatch.setenv("GEOTRACE_RETRIEVAL_BACKEND", "cloud")
        with pytest.raises(retrieval.RetrievalError, match="local, space"):
            retrieval.get_backend()
    finally:
        retrieval.set_backend(None)
