"""Rank reference images by visual similarity to the query image, with MegaLoc.

MegaLoc (Berton & Masone, "MegaLoc: One Retrieval to Place Them All", CVPRW 2025,
https://github.com/gmberton/MegaLoc, MIT) gives one L2-normalised global descriptor per image;
cosine similarity between descriptors ranks the references.

`rank_references` is the only entry point the tools use. It delegates to a backend, so the local
CPU model can later be replaced by a remote one (for example a Hugging Face Space) without
touching the tools. Only the local backend exists for now.
"""

import asyncio
import logging
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


class LocalMegaLoc:
    """MegaLoc on the local CPU (or whatever device torch picks by default)."""

    def __init__(self, cache: EmbeddingCache | None = None):
        self._cache = cache

    @property
    def cache(self) -> EmbeddingCache:
        if self._cache is None:
            self._cache = EmbeddingCache(default_cache_path())
        return self._cache

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

        def embed_all() -> tuple[Any, dict[str, Any]]:
            vectors = _embed([query_bytes], QUERY_DECODE_MAX_SIDE) + _embed([thumbs[i] for i in new_ids])
            query_vec = vectors[0]
            fresh = {image_id: v for image_id, v in zip(new_ids, vectors[1:]) if v is not None}
            if fresh:
                dim = len(next(iter(fresh.values())))
                self.cache.put_many({k: v.astype(np.float32).tobytes() for k, v in fresh.items()}, dim)
            return query_vec, fresh

        query_vec, fresh = await asyncio.to_thread(embed_all)
        if query_vec is None:
            raise ValueError("The query image could not be decoded.")

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


_backend: RetrievalBackend | None = None


def get_backend() -> RetrievalBackend:
    global _backend
    if _backend is None:
        _backend = LocalMegaLoc()
    return _backend


def set_backend(backend: RetrievalBackend | None) -> None:
    """Swap the retrieval backend (for a remote one, or for tests). None restores the default."""
    global _backend
    _backend = backend


async def rank_references(
    query_bytes: bytes, refs: Sequence[MapillaryImage], top_k: int = DEFAULT_TOP_K
) -> list[RankedReference]:
    """The top_k references most similar to the query image, most similar first."""
    return await get_backend().rank(query_bytes, refs, top_k)
