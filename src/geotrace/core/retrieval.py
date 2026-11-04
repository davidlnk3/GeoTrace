"""Rank reference images by visual similarity to the query image, with MegaLoc.

MegaLoc (Berton & Masone, "MegaLoc: One Retrieval to Place Them All", CVPRW 2025,
https://github.com/gmberton/MegaLoc, MIT) gives one L2-normalised global descriptor per image;
cosine similarity between descriptors ranks the references.

`rank_references` is the only entry point the tools use. It delegates to a backend, so the local
CPU model can later be replaced by a remote one (for example a Hugging Face Space) without
touching the tools. Two backends exist: LocalMegaLoc (this machine, default) and SpaceMegaLoc
(the davidlnk3/megaloc-descriptors Space on ZeroGPU), chosen with GEOTRACE_RETRIEVAL_BACKEND.
Both pin the same MegaLoc commit and share one embedding cache.
"""

import asyncio
import logging
import os
import sqlite3
import threading
from collections.abc import Sequence
from contextlib import closing
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from geotrace.core import mapillary
from geotrace.core.images import open_rgb
from geotrace.core.mapillary import MapillaryImage

logger = logging.getLogger(__name__)

# Pinned so the model code torch.hub downloads cannot change under us.
MEGALOC_COMMIT = "5fe0dd697c4a70ba3e23607f6716ab3c606b16db"
MEGALOC_HUB_REPO = f"gmberton/MegaLoc:{MEGALOC_COMMIT}"
MODEL_TAG = f"megaloc-{MEGALOC_COMMIT[:12]}"   # part of the embedding cache key
INPUT_SIZE = 322                               # evaluation preprocessing from the MegaLoc README
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
EMBED_BATCH_SIZE = 16
DEFAULT_TOP_K = 20
# The query is decoded at this size before the 322x322 resize, to avoid working on huge photos.
QUERY_DECODE_MAX_SIDE = 1024


@dataclass(frozen=True)
class RankedReference:
    image: MapillaryImage
    score: float                               # cosine similarity to the query, in [-1, 1]


class RetrievalBackend(Protocol):
    async def rank(
        self, query_bytes: bytes, refs: Sequence[MapillaryImage], top_k: int
    ) -> list[RankedReference]: ...


# ---------------------------------------------------------------------------- embedding cache

def default_cache_path() -> Path:
    from platformdirs import user_cache_dir

    return Path(user_cache_dir("geotrace", appauthor=False)) / "megaloc_embeddings.sqlite3"


class EmbeddingCache:
    """Persistent float32 descriptors in SQLite, keyed by (model tag, Mapillary image id)."""

    def __init__(self, path: Path, model_tag: str = MODEL_TAG):
        self.path = path
        self.model_tag = model_tag
        self._ready = False

    def _connect(self) -> sqlite3.Connection:
        if not self._ready:
            self.path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(self.path, timeout=10)
        if not self._ready:
            conn.execute(
                "CREATE TABLE IF NOT EXISTS embeddings ("
                "model TEXT NOT NULL, image_id TEXT NOT NULL, dim INTEGER NOT NULL, vector BLOB NOT NULL, "
                "PRIMARY KEY (model, image_id))"
            )
            conn.commit()
            self._ready = True
        return conn

    def get_many(self, image_ids: Sequence[str]) -> dict[str, bytes]:
        """Raw float32 bytes for each cached id."""
        out: dict[str, bytes] = {}
        ids = list(dict.fromkeys(image_ids))
        with closing(self._connect()) as conn:
            for start in range(0, len(ids), 500):
                chunk = ids[start:start + 500]
                marks = ",".join("?" * len(chunk))
                rows = conn.execute(
                    f"SELECT image_id, vector FROM embeddings WHERE model = ? AND image_id IN ({marks})",
                    [self.model_tag, *chunk],
                )
                out.update({image_id: vector for image_id, vector in rows})
        return out

    def put_many(self, vectors: dict[str, bytes], dim: int) -> None:
        if not vectors:
            return
        with closing(self._connect()) as conn:
            conn.executemany(
                "INSERT OR REPLACE INTO embeddings (model, image_id, dim, vector) VALUES (?, ?, ?, ?)",
                [(self.model_tag, image_id, dim, vector) for image_id, vector in vectors.items()],
            )
            conn.commit()


# ---------------------------------------------------------------------------- local MegaLoc

_model_lock = threading.Lock()
_model: Any = None


def _load_model() -> Any:
    """Load MegaLoc once, as its README shows: torch.hub.load("gmberton/MegaLoc", "get_trained_model")."""
    global _model
    with _model_lock:
        if _model is None:
            import torch

            # trust_repo=True: the default ("check") may prompt on stdin, which is the MCP channel.
            # torch annotates trust_repo as str, but documents and accepts a bool.
            model = torch.hub.load(
                MEGALOC_HUB_REPO,
                "get_trained_model",
                trust_repo=True,  # pyright: ignore[reportArgumentType]
                skip_validation=True,
            )
            model.eval()
            _model = model
        return _model


def _transform() -> Any:
    import torchvision.transforms as tfm

    return tfm.Compose([
        tfm.ToTensor(),
        tfm.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        tfm.Resize(size=[INPUT_SIZE, INPUT_SIZE], antialias=True),
    ])


def _embed(images: Sequence[bytes], decode_max_side: int | None = None) -> list[Any]:
    """L2-normalised float32 numpy descriptors, one per image; None where the image cannot be decoded."""
    import torch
    import torch.nn.functional as F

    model = _load_model()
    transform = _transform()
    out: list[Any] = [None] * len(images)
    tensors: list[tuple[int, Any]] = []
    for i, data in enumerate(images):
        try:
            tensors.append((i, transform(open_rgb(data, decode_max_side))))
        except Exception as exc:  # noqa: BLE001 - a corrupt thumbnail just drops out of the ranking
            logger.debug("skipping undecodable image %d: %s", i, exc)
    with torch.inference_mode():
        for start in range(0, len(tensors), EMBED_BATCH_SIZE):
            chunk = tensors[start:start + EMBED_BATCH_SIZE]
            batch = torch.stack([t for _, t in chunk])
            desc = F.normalize(model(batch).float(), dim=-1).cpu().numpy()
            for (i, _), vec in zip(chunk, desc):
                out[i] = vec
    return out


class RetrievalError(ValueError):
    """Raised when descriptors cannot be computed. The message is safe to show the model."""


class _MegaLocBackend:
    """Ranking shared by every MegaLoc backend: the cache lookup, thumbnail download and scoring.

    Subclasses only compute descriptors, in `_embed`.
    """

    def __init__(self, cache: EmbeddingCache | None = None):
        self._cache = cache

    @property
    def cache(self) -> EmbeddingCache:
        if self._cache is None:
            self._cache = EmbeddingCache(default_cache_path())
        return self._cache

    async def _embed(self, query_bytes: bytes, refs: Sequence[bytes]) -> tuple[Any, list[Any]]:
        """(query descriptor, one descriptor per reference); None where an image cannot be decoded."""
        raise NotImplementedError

    async def rank(
        self, query_bytes: bytes, refs: Sequence[MapillaryImage], top_k: int
    ) -> list[RankedReference]:
        import numpy as np

        if not refs or top_k <= 0:
            return []
        cached = await asyncio.to_thread(self.cache.get_many, [r.id for r in refs])
        missing = [r for r in refs if r.id not in cached]
        thumbs = await mapillary.download_thumbnails(missing, 256)
        new_ids = list(thumbs)

        query_vec, new_vecs = await self._embed(query_bytes, [thumbs[i] for i in new_ids])
        if query_vec is None:
            raise RetrievalError("The query image could not be decoded.")
        fresh = {
            image_id: np.asarray(v, dtype=np.float32)
            for image_id, v in zip(new_ids, new_vecs) if v is not None
        }
        if fresh:
            dim = len(next(iter(fresh.values())))
            await asyncio.to_thread(
                self.cache.put_many, {k: v.tobytes() for k, v in fresh.items()}, dim
            )

        query_vec = np.asarray(query_vec, dtype=np.float32)
        ranked: list[RankedReference] = []
        for ref in refs:
            if ref.id in fresh:
                vec = fresh[ref.id]
            elif ref.id in cached:
                vec = np.frombuffer(cached[ref.id], dtype=np.float32)
            else:
                continue                       # thumbnail missing or undecodable
            if vec.shape != query_vec.shape:
                continue
            ranked.append(RankedReference(image=ref, score=float(np.dot(query_vec, vec))))
        ranked.sort(key=lambda r: r.score, reverse=True)
        return ranked[:top_k]


class LocalMegaLoc(_MegaLocBackend):
    """MegaLoc on the local CPU (or whatever device torch picks by default)."""

    async def _embed(self, query_bytes: bytes, refs: Sequence[bytes]) -> tuple[Any, list[Any]]:
        def run() -> tuple[Any, list[Any]]:
            return _embed([query_bytes], QUERY_DECODE_MAX_SIDE)[0], _embed(refs)

        return await asyncio.to_thread(run)


# ---------------------------------------------------------------------------- Space MegaLoc

DEFAULT_SPACE_ID = "davidlnk3/megaloc-descriptors"
SPACE_ENV = "GEOTRACE_MEGALOC_SPACE"
SPACE_MAX_IMAGES = 256             # per call, enforced by the Space
SPACE_TIMEOUT_S = 180.0
QUERY_UPLOAD_QUALITY = 90


def _to_jpeg(data: bytes, max_side: int | None) -> bytes | None:
    """Upright RGB JPEG of the image (EXIF orientation applied, HEIC decoded), or None if unreadable."""
    import io

    try:
        rgb = open_rgb(data, max_side)
    except Exception as exc:  # noqa: BLE001
        logger.debug("skipping undecodable image: %s", exc)
        return None
    buf = io.BytesIO()
    rgb.save(buf, format="JPEG", quality=QUERY_UPLOAD_QUALITY)
    return buf.getvalue()


def _decode_vectors(payload: Any, expected: int) -> list[Any]:
    """Descriptors from the Space's JSON answer (base64 float16), as float32 numpy arrays."""
    import base64
    import json

    import numpy as np

    if isinstance(payload, str):
        payload = json.loads(payload)
    if not isinstance(payload, dict) or not isinstance(payload.get("vectors"), list):
        raise RetrievalError("The MegaLoc Space returned an unexpected answer.")
    if payload.get("model") != MODEL_TAG:
        raise RetrievalError(
            f"The MegaLoc Space runs model {payload.get('model')!r}, but GeoTrace expects {MODEL_TAG!r}. "
            "Update GeoTrace or the Space so both pin the same MegaLoc commit."
        )
    vectors = payload["vectors"]
    if len(vectors) != expected:
        raise RetrievalError("The MegaLoc Space returned the wrong number of descriptors.")
    dtype = np.dtype("<f2") if payload.get("dtype") == "float16" else np.dtype("<f4")
    return [
        None if v is None else np.frombuffer(base64.b64decode(v), dtype=dtype).astype(np.float32)
        for v in vectors
    ]


class SpaceMegaLoc(_MegaLocBackend):
    """MegaLoc on a Hugging Face Space (ZeroGPU), called with gradio_client.

    Only the images whose descriptors are not cached are uploaded: the query, re-encoded as a JPEG
    of at most QUERY_DECODE_MAX_SIDE px, and the 256 px Mapillary thumbnails. HF_TOKEN, if set, is
    sent so the calls use the account's ZeroGPU quota instead of the small anonymous one.
    """

    def __init__(
        self,
        space_id: str | None = None,
        cache: EmbeddingCache | None = None,
        client_factory: Any = None,
    ):
        super().__init__(cache)
        self.space_id = space_id or os.environ.get(SPACE_ENV, "").strip() or DEFAULT_SPACE_ID
        self._client_factory = client_factory
        self._client: Any = None
        self._client_lock = threading.Lock()

    def _get_client(self) -> Any:
        with self._client_lock:
            if self._client is None:
                if self._client_factory is not None:
                    self._client = self._client_factory(self.space_id)
                else:
                    from gradio_client import Client

                    self._client = Client(
                        self.space_id,
                        token=os.environ.get("HF_TOKEN") or None,
                        verbose=False,
                        httpx_kwargs={"timeout": SPACE_TIMEOUT_S},
                    )
            return self._client

    def _call(self, images: list[bytes]) -> list[Any]:
        """Send the images in chunks of SPACE_MAX_IMAGES; one descriptor (or None) per image."""
        import tempfile
        from pathlib import Path as _Path

        from gradio_client import handle_file

        client = self._get_client()
        out: list[Any] = []
        with tempfile.TemporaryDirectory(prefix="geotrace-megaloc-") as tmp:
            for start in range(0, len(images), SPACE_MAX_IMAGES):
                chunk = images[start:start + SPACE_MAX_IMAGES]
                paths = []
                for i, data in enumerate(chunk):
                    path = _Path(tmp) / f"{start + i}.jpg"
                    path.write_bytes(data)
                    paths.append(handle_file(str(path)))
                try:
                    payload = client.predict(images=paths, api_name="/embed")
                except Exception as exc:  # gradio_client raises many types
                    raise RetrievalError(
                        f"The MegaLoc Space {self.space_id} failed: {exc}. If it is asleep or out of "
                        "ZeroGPU quota, retry in a minute, set HF_TOKEN, or set "
                        "GEOTRACE_RETRIEVAL_BACKEND=local to run MegaLoc on this machine."
                    ) from exc
                out.extend(_decode_vectors(payload, len(chunk)))
        return out

    async def _embed(self, query_bytes: bytes, refs: Sequence[bytes]) -> tuple[Any, list[Any]]:
        def run() -> tuple[Any, list[Any]]:
            query = _to_jpeg(query_bytes, QUERY_DECODE_MAX_SIDE)
            if query is None:
                return None, [None] * len(refs)
            # Thumbnails are already small JPEGs; undecodable ones get None from the Space.
            vectors = self._call([query, *refs])
            return vectors[0], vectors[1:]

        return await asyncio.to_thread(run)


# ---------------------------------------------------------------------------- selection

BACKEND_ENV = "GEOTRACE_RETRIEVAL_BACKEND"
BACKENDS = {"local": LocalMegaLoc, "space": SpaceMegaLoc}

_backend: RetrievalBackend | None = None


def get_backend() -> RetrievalBackend:
    """The backend named by GEOTRACE_RETRIEVAL_BACKEND: "local" (default) or "space"."""
    global _backend
    if _backend is None:
        name = os.environ.get(BACKEND_ENV, "local").strip().lower() or "local"
        if name not in BACKENDS:
            raise RetrievalError(
                f"{BACKEND_ENV} must be one of: {', '.join(BACKENDS)} (got {name!r})."
            )
        backend: RetrievalBackend = BACKENDS[name]()
        _backend = backend
        return backend
    return _backend


def set_backend(backend: RetrievalBackend | None) -> None:
    """Swap the retrieval backend (for tests, or another remote one). None restores the default."""
    global _backend
    _backend = backend


async def rank_references(
    query_bytes: bytes, refs: Sequence[MapillaryImage], top_k: int = DEFAULT_TOP_K
) -> list[RankedReference]:
    """The top_k references most similar to the query image, most similar first."""
    return await get_backend().rank(query_bytes, refs, top_k)
