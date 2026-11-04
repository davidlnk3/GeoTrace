---
title: MegaLoc Descriptors
emoji: 📍
colorFrom: blue
colorTo: green
sdk: gradio
sdk_version: 6.31.0
app_file: app.py
python_version: "3.12"
short_description: MegaLoc place-recognition descriptors for GeoTrace
license: mit
models:
  - gberton/MegaLoc
---

# MegaLoc descriptors

Computes global visual place recognition descriptors with
[MegaLoc](https://github.com/gmberton/MegaLoc) (Berton & Masone, *MegaLoc: One Retrieval to Place
Them All*, CVPR Workshops 2025). It is the remote retrieval backend of
[GeoTrace](https://github.com/davidlnk3/GeoTrace)'s `verify_candidate_locations` tool.

## API

`/embed` takes up to 256 image files and returns

```json
{"model": "megaloc-5fe0dd697c4a", "dim": 8448, "dtype": "float16", "vectors": ["<base64>", null, ...]}
```

Each vector is the base64 of little-endian float16 values, L2-normalised, in the order of the inputs
(`null` for an image that could not be decoded). Cosine similarity is the dot product of two vectors.

```python
from gradio_client import Client, handle_file

client = Client("davidlnk3/megaloc-descriptors")
result = client.predict(images=[handle_file("a.jpg"), handle_file("b.jpg")], api_name="/embed")
```

Preprocessing follows the MegaLoc README: RGB, ImageNet normalisation, resize to 322×322.

## Licence

MegaLoc code and weights: MIT, by Gabriele Berton and Carlo Masone. The model code is loaded with
`torch.hub` from `gmberton/MegaLoc` at commit `5fe0dd697c4a70ba3e23607f6716ab3c606b16db`, unmodified;
the weights come from [gberton/MegaLoc](https://huggingface.co/gberton/MegaLoc). This Space's code is
MIT, like GeoTrace.
