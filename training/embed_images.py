"""Compute one OpenCLIP embedding per labelled image, for De Boer tarballs and DEW folders.

The De Boer tarball is ~10.6 GB compressed and this machine has little free disk, so it is
never extracted: `tarfile` reads it as a gzip stream ('r|gz') and each JPEG is embedded and
dropped. Only images named in the authors' image_data.json are kept.

Backbone matches the paper's reference implementation exactly
(CANAL-amsterdam/dating-images, 1.Embeddings.ipynb): ViT-B-32-quickgelu / laion400m_e32,
raw `encode_image` output with NO L2 normalisation.
"""
import argparse
import io
import json
import sys
import tarfile
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

MODEL = "ViT-B-32-quickgelu"
PRETRAINED = "laion400m_e32"
Image.MAX_IMAGE_PIXELS = 200_000_000  # inputs are downloaded data; cap decompression bombs


def key_of(path: str) -> str:
    """Identify an image by its last two path parts: '<scene>/<file>.jpg'."""
    parts = [p for p in path.replace("\\", "/").split("/") if p not in ("", ".", "..")]
    return "/".join(parts[-2:])


class Embedder:
    def __init__(self, batch_size: int):
        import open_clip

        self.device = "mps" if torch.backends.mps.is_available() else "cpu"
        model, _, preprocess = open_clip.create_model_and_transforms(MODEL, pretrained=PRETRAINED)
        model.eval().to(self.device)
        self.model = model
        self.preprocess = preprocess
        self.dim = model.visual.output_dim
        self.batch_size = batch_size
        print(f"{MODEL}/{PRETRAINED} on {self.device}, dim={self.dim}", flush=True)

    def encode(self, tensors: list[torch.Tensor]) -> np.ndarray:
        x = torch.stack(tensors).to(self.device)
        with torch.inference_mode():
            return self.model.encode_image(x).float().cpu().numpy()


def run(embedder: Embedder, items, total: int, out_path: str) -> int:
    """items yields (index, image_bytes). Writes an (total, dim) float32 matrix."""
    embeddings = np.zeros((total, embedder.dim), dtype=np.float32)
    found = np.zeros(total, dtype=bool)
    batch_t: list[torch.Tensor] = []
    batch_i: list[int] = []
    seen = 0
    start = time.time()

    def flush():
        if not batch_t:
            return
        embeddings[batch_i] = embedder.encode(batch_t)
        found[batch_i] = True
        batch_t.clear()
        batch_i.clear()

    for idx, raw in items:
        try:
            with Image.open(io.BytesIO(raw)) as img:
                batch_t.append(embedder.preprocess(img.convert("RGB")))
        except Exception as exc:  # noqa: BLE001 - one unreadable JPEG must not end a 40k-image run
            print(f"  skip #{idx}: {type(exc).__name__}: {exc}", flush=True)
            continue
        batch_i.append(idx)
        seen += 1
        if len(batch_t) >= embedder.batch_size:
            flush()
            if seen % (embedder.batch_size * 20) == 0:
                print(f"  {seen}/{total}  {seen / (time.time() - start):.1f} img/s", flush=True)
    flush()

    n = int(found.sum())
    print(f"embedded {n}/{total} in {time.time() - start:.0f}s", flush=True)
    np.savez_compressed(out_path, embeddings=embeddings, found=found)
    print(f"wrote {out_path}", flush=True)
    return 0 if n else 1


def from_tar(tar_path: str, labels_path: str):
    with open(labels_path) as fh:
        records = json.load(fh)
    wanted = {key_of(r[0]): i for i, r in enumerate(records)}
    print(f"{len(records)} labelled images, {len(wanted)} unique keys", flush=True)
    seen_idx: set[int] = set()

    def gen():
        with tarfile.open(tar_path, "r|gz") as tar:
            for member in tar:
                if not member.isfile():
                    continue
                idx = wanted.get(key_of(member.name))
                if idx is None or idx in seen_idx:
                    continue
                fh = tar.extractfile(member)
                if fh is None:
                    continue
                seen_idx.add(idx)
                yield idx, fh.read()

    return gen(), len(records)


def from_folder(folder: str, manifest_path: str):
    """manifest: list of {"file": <name under folder>, "year": int}."""
    with open(manifest_path) as fh:
        rows = json.load(fh)
    base = Path(folder)

    def gen():
        for i, row in enumerate(rows):
            p = base / row["file"]
            if p.is_file():
                yield i, p.read_bytes()

    return gen(), len(rows)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tar", help="De Boer scene_detection.tar.gz")
    ap.add_argument("--labels", help="authors' image_data.json (with --tar)")
    ap.add_argument("--folder", help="folder of images (DEW)")
    ap.add_argument("--manifest", help="json list of {file, year} (with --folder)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--batch", type=int, default=64)
    args = ap.parse_args()

    if args.tar:
        items, total = from_tar(args.tar, args.labels)
    elif args.folder:
        items, total = from_folder(args.folder, args.manifest)
    else:
        ap.error("pass --tar or --folder")

    return run(Embedder(args.batch), items, total, args.out)


if __name__ == "__main__":
    sys.exit(main())
