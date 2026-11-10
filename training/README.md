# Training the photographic-style year classifier

This is the pipeline behind [`davidlnk3/geotrace-photo-year`](https://huggingface.co/davidlnk3/geotrace-photo-year),
the capture-year estimate that `time_from_objects` returns alongside its object intervals.

It reproduces **["Blind Dates: Examining the Expression of Temporality in Historical Photographs"](https://arxiv.org/abs/2310.06633)**
(Barancová, Wevers & van Noord, CHR 2023): a multinomial logistic regression over one class per
year, on top of **frozen** OpenCLIP image embeddings. Nothing in the backbone is fine-tuned, so
training is CPU/GPU-cheap — the cost is embedding the images once.

The paper itself reports no hyperparameters. These come from the authors' own code,
[CANAL-amsterdam/dating-images](https://github.com/CANAL-amsterdam/dating-images):

| | |
|---|---|
| Backbone | `ViT-B-32-quickgelu`, pretrained `laion400m_e32`, via `open_clip` |
| Embedding | raw `encode_image` output, **no L2 normalisation**, 512-d |
| Head | `LogisticRegression(random_state=0, max_iter=5000)` |
| Classes | one per year, 1950–1999 (50 classes) |
| Split | the authors' frozen stratified 80/20 split, read from their `data/image_data.json` |

## Data

- **De Boer Scene Detection** — 39,866 dated Dutch press photographs, the paper's own corpus.
  [Zenodo 7137452](https://zenodo.org/record/7137452), `scene_detection.tar.gz` (10.6 GB).
  Labels and the train/test split come from `data/image_data.json` in the authors' repo, so the
  test set here is exactly theirs and the MAE is comparable to their 6.65 years.
- **DEW** — 500 extra Flickr photos, added to **train only** so the test set stays untouched.
  Licences vary per photo and many are not open, so the manifest records each one and no image is
  redistributed. DEW ships a CSV of URLs, not images: download `meta.csv` from
  [doi:10.22000/43](https://doi.org/10.22000/43) (the RADAR page asks you to accept its terms, so
  this one file has to be fetched by hand) and `dew_fetch.py` pulls the photos from Flickr.

## Running it

```bash
# 1. embed the De Boer corpus, streaming straight from the tarball (never extracted: ~10 GB)
python training/embed_images.py \
  --tar scene_detection.tar.gz --labels image_data.json --out deboer_emb.npz

# 2. sample and download 500 DEW photos, spread evenly over 1950-1999
python training/dew_fetch.py --meta meta.csv --out dew/ --n 500
python training/embed_images.py \
  --folder dew/images --manifest dew/manifest.json --out dew_emb.npz

# 3. fit the head and export it
python training/train_dating.py \
  --labels image_data.json --deboer-emb deboer_emb.npz \
  --dew-manifest dew/manifest.json --dew-emb dew_emb.npz --out-dir out/
```

Step 3 fits a De Boer-only baseline first and then the final model, so the effect of the DEW
images is measured rather than assumed. Both sets of numbers land in `out/config.json`.

Needs `torch`, `open_clip_torch`, `scikit-learn`, `safetensors` and `pillow`. `scikit-learn` is a
**training-only** dependency: the head is exported as a weight matrix and a bias vector in
safetensors, and `core/photo_dating.py` serves it with a single numpy matmul. That keeps
scikit-learn out of the server and avoids unpickling a model at load time.
