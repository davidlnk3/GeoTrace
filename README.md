<p align="center">
  <img src="assets/geotrace-banner.png" alt="GeoTrace — Geolocate • Chronolocate" width="100%">
</p>

<div align="center">

### MCP for open source content verification from images

**Get the position and the time of any picture**
</div>

GeoTrace allows you to find the where and the when an image was taken. The main purpose of this work is educational and, in case it can, to help journalists and OSINT experts in the research of criminal activity and similar. My interest in OSINT started with Sofia Santos and bellingcat challenges, whose I share their vision of revealing human rights violations all around the world in order to concern people and authorities about them. Therefore, this project aims to provide a tool to people wanting to collaborate in this tasks.

## Tools

### `extract_metadata`

Extracts EXIF metadata, camera data and GPS from an image.

- `image`: URL or local path of the image.

### `view_image`

Returns the image so the model can describe its visual clues. The image is downscaled so its longest side is at most 1568 px.

- `image`: URL or local path of the image.
- `grid` (optional): if `true`, overlays a labelled 4×4 grid. Columns are `A`–`D` from left to right and rows are `1`–`4` from top to bottom, so `C2` is the third column and second row. The grid is drawn on a copy of the image and is not part of the evidence. Default `false`.

### `zoom_image`

Returns one region of the image, cropped from the original file at full resolution, so the model can read small details that the downscaled `view_image` copy loses.

- `image`: URL or local path of the image.
- `cell`: a grid cell from `view_image(grid=True)`, for example `"C2"`.
- `region`: `[x1, y1, x2, y2]` as fractions (0–1) of the width and height, with `(0, 0)` the top-left corner. Each side must be at least 2% of the image.

Give exactly one of `cell` or `region`. Coordinates refer to the image after its EXIF orientation is applied, which is how `view_image` shows it.

The crop is fitted to a longest side of 1568 px. Larger crops are shrunk, and smaller crops are enlarged by up to 4× with LANCZOS interpolation. An enlarged crop has bigger pixels but no new detail, so text that is still unreadable must be reported as unreadable, not guessed. AI super-resolution is never used, because it invents detail.

Returns the cropped image and a JSON text with `region`, `cell` (if given), `source_pixels` (the `[left, top, right, bottom]` box in the original image), `source_size`, `image_size`, `scale` and `upscaled`.

#### Zoom workflow

1. Call `view_image(image, grid=True)` to see the whole picture with cell labels.
2. Act on one cell:
   - `zoom_image(image, cell="C2")` to read or check a detail, such as a sign, a shop front or a building plaque.
   - `extract_text(image, cell="C2")` to transcribe the text in that cell, read from the full-resolution original.
   - `reverse_search_image(image, cell="C2")` to find where a landmark, building, logo or distinctive sign appears online.
3. For an object that crosses cell borders, use `region` with fractions taken from the grid. For example, columns B–C and row 2 give `[0.25, 0.25, 0.75, 0.5]`.

### `reverse_search_image`

Runs a reverse image search on the whole image or on one region of it, to find pages and images showing the same landmark, building, logo or distinctive sign. **It must never be used on people or faces.**

- `image`: URL or local path of the image.
- `cell` or `region`: as in `zoom_image`. Give exactly one. Use `region=[0, 0, 1, 1]` to search the whole image.
- `backend` (optional): `serpapi` (Google Lens through SerpApi, the default) or `cloud_vision` (Google Cloud Vision Web Detection).

Returns:

- `backend`: the backend that ran the search.
- `best_guess`: labels or entity names. For SerpApi these are Google Lens's related searches; for Cloud Vision they are the best-guess labels followed by web entities.
- `pages`: pages where the image or similar ones appear, each with `url`, `title` and `source`.
- `exact_matches`: URLs of identical images.
- `visual_matches`: similar images, each with `url`, `title`, `source` and `thumbnail`.
- `total_results`, `truncated`: each list holds at most 15 items, and `truncated` says whether any were cut.
- `note`: set when the tool fell back to another backend.
- `region`, `source_pixels`, `cell`: the region that was searched.
- `cached`: whether the result came from the cache.

Results are leads to verify, not proof of a location.

Choosing a good crop:

- Include the whole object, with a little context around it.
- Pick something distinctive, such as a named facade, a unique sign or a logo. A generic lamp post, tree or road gives useless matches.
- Make sure the region has enough original resolution. It must be at least **200 px per side in the original image**, and smaller regions are refused. Use `zoom_image` first to check.

Safeguards:

- The tool description tells the model never to search people.
- OpenCV's Haar face detector runs on every crop. The search is refused if one face covers more than 10% of the crop. This is a safeguard, not a guarantee: it misses faces that are small, turned away, hidden or blurred.
- Results are cached in memory by the SHA-256 of the crop and the backend, so the same crop is never searched (and billed) twice while the server runs.

Requires the `reverse-search` extra, which installs `opencv-python-headless` (4.x) for the face check: `uv sync --extra reverse-search`. `zoom_image` and the grid do not need it.

#### Configuration

Set the keys in the `"env"` block of the GeoTrace entry in `claude_desktop_config.json`, then restart Claude Desktop:

```json
{
  "mcpServers": {
    "geotrace": {
      "command": "uv",
      "args": ["--directory", "/path/to/GeoTrace", "run", "geotrace"],
      "env": {
        "SERPAPI_API_KEY": "your-serpapi-key",
        "GOOGLE_VISION_API_KEY": "your-google-cloud-key"
      }
    }
  }
}
```

- **SerpApi (default):** create an account at [serpapi.com](https://serpapi.com) and copy the API key into `SERPAPI_API_KEY`. The crop is uploaded to SerpApi's own image endpoint (`POST https://serpapi.com/image`, at most 500 KB, re-encoded smaller if needed). The search then uses the returned `image_id`, which SerpApi expires after 10 minutes. No other image host is used. SerpApi does not document how long it keeps uploaded images or whether they can be deleted.
- **Google Cloud Vision (optional):** in a Google Cloud project, enable the Cloud Vision API, turn on billing and create an API key, ideally restricted to the Vision API. Put it in `GOOGLE_VISION_API_KEY`. The crop is sent inline in the request.

You can set either key or both. If `backend="serpapi"` is requested but only `GOOGLE_VISION_API_KEY` is set, the tool uses Cloud Vision and says so in `backend` and `note`. If neither is set, the tool returns an error listing both variables.

> **About SerpApi.** SerpApi is a third-party paid service that scrapes Google results. Each search uses credits from your plan. Google sued SerpApi in December 2025, and the case was still pending at last check. The service could stop working at any time. You decide whether to use it. Google Cloud Vision Web Detection is the official alternative.

### `relative_sun_position_from_shadows`

Detects object-shadow pairs in an image and computes the sun elevation and the shadow direction of each pair. The camera is calibrated with GeoCalib, and the sun elevation is computed from that calibration. Requires the `calibration` extra: `pip install -e ".[calibration]"`.

- `image`: URL or local path of the image.

Returns:

- `camera`: the calibration (`roll_deg`, `pitch_deg`, `vertical_fov_deg`, `pitch_uncertainty_deg`, `vertical_fov_uncertainty_deg`).
- `num_pairs`, `pairs`: each pair has `pair_score` (detection confidence), `shadow_direction_deg` (on the ground, clockwise seen from above from the camera's forward direction: 0 = away from the camera, 90 = right) and `sun_elevation_deg`.
- `skipped`: pairs that could not be computed, with the reason (for example, a base or tip above the horizon).

### `relative_sun_position_from_outdoor_lighting`

Estimates the sun's altitude and its azimuth relative to the camera from a single outdoor image, with a DINOv2-small model trained on crops where the sun is outside the frame. It uses the centre square of the image.

- `image`: URL or local path of the image.

Returns:

- `sun_altitude_deg`: sun elevation above the horizon, 0 to 90.
- `sun_azimuth_relative_camera_deg`: sun direction clockwise from the camera's forward direction, in [0, 360). 0 = straight ahead, 90 = right, 180 = behind.

The model reads the sun from scene cues (shadows, lighting, sky brightness), so it is an estimate. On the validation set of the training project, the median errors were 1.6° for altitude and 5.6° for azimuth, with a long tail of large azimuth errors.

Requires `torch` and `transformers`, which are not in the default dependencies. Install them in the GeoTrace environment with `pip install torch transformers`. The model weights are in the repository at `models/sun/best.pt`, so the tool needs no external files.

### `time_from_sun_position`

Given a location and the sun's position, returns the dates and times (UTC) when the sun was in that position.

- `latitude`, `longitude`: decimal degrees.
- `sun_elevation`: degrees above the horizon, greater than 0.
- `sun_azimuth`: degrees clockwise from north. A shadow points opposite the sun, so `sun_azimuth = shadow_direction + 180` (mod 360).
- `year` (optional): year to search. Defaults to the current year.
- `tolerance` (optional): maximum angular error in degrees. Defaults to 1.
- `max_results` (optional): maximum number of candidates. Defaults to 10.

Returns the candidates sorted by error, with the date and time in UTC.

### `location_from_sun_position`

Given the date and time of a photo and the sun's position, returns the locations where the sun was in that position.

- `date`: UTC date, `YYYY-MM-DD`.
- `time`: UTC time, `HH:MM` or `HH:MM:SS`.
- `sun_elevation`: degrees above the horizon, greater than 0.
- `sun_azimuth`: degrees clockwise from north.

Usually two mirror-image solutions exist; both are returned, each with latitude and longitude.

### `time_from_objects`

Gives the interval in which each named object was built, from its likely build year until today, with EVA02-CLIP time probing as in [arXiv:2510.19559](https://arxiv.org/abs/2510.19559). It helps narrow when the photo can have been taken: it cannot be older than the most recent object in it.

- `image`: URL or local path of the image.
- `objects` (required): a list of object names to date, for example `["bicycle", "street lamp"]`. It must contain at least one name.

Returns:

- `current_year`, `image_taken_not_before_year`: the latest `built_from_year` among the objects.
- `objects`: for each object, `built_from_year`, `built_until_year` (today), and `tolerance_years`.

`built_from_year` is the most likely build year minus the paper's accuracy tolerance for that year (20 years at 1700, 5 years at 2024). The paper scores whole images with prompts that name no object; adding the object name to the prompt, and the interval rule, are this project's extensions and are not evaluated in the paper.

Requires `torch`, `transformers` and `open_clip_torch`: `pip install torch open_clip_torch`. The EVA02-CLIP weights (`EVA02-L-14-336`, `merged2b_s6b_b61k`) are downloaded on first use.

### `overpass_query`

Runs an Overpass QL query on the OpenStreetMap Overpass API inside a bounding box. It tries `https://overpass-api.de/api/interpreter` first, then falls back in order to `https://overpass.kumi.systems/api/interpreter`, `https://overpass.private.coffee/api/interpreter`, `https://maps.mail.ru/osm/tools/overpass/api/interpreter` and `https://ovp.dresel.systems/api/interpreter` if an endpoint fails (timeout, connection error, busy server or any other error).

- `query`: Overpass QL statements. Write the body only; the tool adds `[out:json]`, the timeout and `out body;`. Every filter must use the `{{bbox}}` placeholder, for example `node["amenity"="cafe"]({{bbox}});`. Queries without it are rejected, so the box limit always applies.
- `south`, `west`, `north`, `east`: the box in decimal degrees. It must not cross the antimeridian.
- `timeout` (optional): server-side timeout in seconds. Default 25, maximum 90.
- `max_results` (optional): maximum number of elements returned. Default 100, maximum 1000.
- `rarity` (optional): how common the searched feature is. It sets the largest allowed box: `rare` 2500 km², `medium` 400 km², `common` 100 km² (default), `very_common` 10 km². Larger boxes are rejected.

Returns `bbox_area_km2`, `count_total`, `count_returned`, `truncated` and `elements` (OSM JSON). Overpass runtime errors, such as a server-side timeout, are reported in `remark`.
### `location_guesser_yfcc`

Estimates where an image was probably taken, using PLONK ([arXiv:2412.06781](https://arxiv.org/abs/2412.06781)) with the `nicolas-dufour/PLONK_YFCC` model. The model draws many candidate locations and they are summarised as regions.

- `image`: URL or local path of the image.
- `num_samples` (optional): how many candidate locations to draw, from 20 to 500. Default 100. More gives steadier regions but costs more time.

Returns:

- `regions`: sorted by probability. Each has `name` (nearest place and admin region), `probability`, `center` (`latitude`, `longitude`), `radius_km` and `country_code`. `regions[0]` is the most probable area, and its centre is the most probable point.
- `unassigned_probability`: the share of samples outside the returned regions.
- `by_country`: each country code's share of all samples.
- `area_90_km2`: area of the fewest 1° cells that hold 90% of the samples.
- `localizability`: `high` (up to 10,000 km²), `medium` (up to 500,000 km²) or `low`, from `area_90_km2`.
- `num_samples`, `model`.

Regions are formed by greedy clustering: each takes the samples within 100 km of the sample with the most neighbours, and its radius is the 90th percentile of its members' distances from the centre. A region needs at least 5% of the samples. The thresholds and rules are this project's choices, not the paper's.

This is an estimate from visual content, not a verified location. The first call downloads the model weights and the DINOv2 code, and it is slow on a CPU; each sample costs about a second.

Requires `torch`, `transformers`, `torchvision`, `diff-plonk` and `reverse_geocoder`: `pip install diff-plonk reverse_geocoder`. The `diff-plonk` package pins `scipy==1.13.1`, which does not build on Python 3.15. There, install it with `--no-deps` and add `scipy`, `geoopt`, `torchdiffeq`, `einops` and `accelerate`.

### `location_guesser_inat`

The same as `location_guesser_yfcc`, but with the `nicolas-dufour/PLONK_iNaturalist` model, trained on iNaturalist. It suits nature photos (plants, animals, landscapes) and is less suited to general street or city photos, which `location_guesser_yfcc` (YFCC) covers better. It takes the same arguments and returns the same fields, with `model` set to `nicolas-dufour/PLONK_iNaturalist`.

### `location_guesser_osv`

The same as `location_guesser_yfcc`, but with the `nicolas-dufour/PLONK_OSV_5M` model, trained on OSV-5M, a street-level dataset. It suits photos of roads, streets, buildings and signs. This model uses a StreetCLIP encoder (`geolocal/StreetCLIP`) instead of DINOv2, so the first call also downloads that encoder. It takes the same arguments and returns the same fields, with `model` set to `nicolas-dufour/PLONK_OSV_5M`.

### `extract_text`

Detects and reads text in an image with [EasyOCR](https://github.com/JaidedAI/EasyOCR): signs, shop names, street names, bus stop names, posters, house numbers.

- `image`: URL or local path of the image.
- `languages` (optional): EasyOCR language codes. The default is common Latin-script languages (`en`, `es`, `fr`, `de`, `it`, `pt`, `nl`). Latin-script languages can be combined freely. Other scripts can only be combined with English, and Arabic, Cyrillic, Devanagari and Bengali also with languages of the same script. Examples: `["ru", "en"]`, `["ar", "en"]`, `["ja", "en"]`, `["ch_sim", "en"]`. An invalid combination is rejected with a message saying which languages to split into separate calls.
- `min_confidence` (optional): detections below this confidence are dropped. Default 0.3.
- `cell` or `region` (optional): as in `zoom_image`, to read only one place of the image. Give at most one; omit both to read the whole image. The region is cropped from the full-resolution original before OCR, so small or distant text is often read better than in a whole-image call.
- `rotated_text` (optional): also read each text box rotated by 90°, 180° and 270°, for vertical or upside-down text. Default `false`: in tests, the rotated readings often replaced correct upright text with misreadings (`STOP` read upside down as `d01S`).

Returns `image_size` (of the original image), `languages`, `blocks`, `total_blocks`, `truncated`, `scale`, `upscaled`, `retried` and `rotated_text`, plus `region`, `source_pixels` (the `[left, top, right, bottom]` box read) and `cell` when a region was given. Each block has `text`, `confidence`, `bbox` (`x1, y1, x2, y2` in pixels of the original image, also for a region) and `relative_size` (block area / area read, i.e. the image or the region). Blocks are sorted by `confidence × √relative_size`, so large sign text comes before small print, and at most 50 are returned. Words on the same line (similar height and baseline, small gap) are merged, so `VIA` + `ROMA` becomes `VIA ROMA`.

The image (or region) is OCR'd at its native resolution, downscaled only when its longest side is above 3000 px. If nothing is found, it is read once more enlarged with LANCZOS (at least 2×, towards a longest side of 1568 px, at most 4× and 3000 px); `retried` and `scale` report it, and a note warns that such readings are near the limit of the pixels. Enlarging is only a fallback: on small text (8–18 px tall) EasyOCR read as well or better at native resolution, and text under about 8 px tall in the original is rarely readable at all. OCR can misread characters (0/O, 1/l, accents), so check the text against the image.

Requires the `ocr` extra, which installs EasyOCR and PyTorch:

```bash
uv sync --extra ocr
# or: pip install -e ".[ocr]"
```

Without it the server still starts, and `extract_text` returns an error explaining how to install it. The first call for each language set downloads the EasyOCR models (to `~/.EasyOCR/`) and is slow. Later calls reuse the models already in memory.

### `find_named_features`

Searches OpenStreetMap for features (nodes, ways and relations) whose name contains a text, inside a bounding box. It needs no extra dependencies.

- `name`: the text to search, for example a shop name read on a sign. It is matched literally (special characters are escaped), case-insensitively, ignoring accents on common Latin letters and with collapsed whitespace, as part of the name: `Farmacia Roma` finds `Antica Farmacia Roma`.
- `bbox`: `{"south": …, "west": …, "north": …, "east": …}` in decimal degrees, at most 400 km², not crossing the antimeridian.
- `tags` (optional): exact tag filters, for example `{"amenity": "pharmacy"}`. Use `"*"` to only require the key.

It looks in `name`, `name:*` (e.g. `name:en`), `alt_name`, `old_name` and `official_name`. It first runs an `out count` query, and then fetches the matches only if there are any. Returns `total_found`, `truncated` and up to 25 `results` nearest to the box centre, each with `name`, `matched_field`, `matched_value`, `osm_type`, `osm_id`, `osm_url`, `lat`, `lon` (the centre for ways and relations), `distance_to_bbox_center_m` and the main tags. If nothing matches, `results` is empty and a `note` suggests a larger box or a shorter part of the name.

### From text in the photo to a place

1. `extract_text` reads the text in the photo.
2. The model checks the texts against the image and picks the ones that name places: shops, streets, stops, stations, businesses.
3. `find_named_features` searches each of them in OpenStreetMap inside the candidate area, for example a region from `location_guesser_osv`.

### `verify_candidate_locations`

Checks candidate locations for a street-level photo against nearby [Mapillary](https://www.mapillary.com) photos. For each candidate it searches the Mapillary photos within `radius_m`, ranks them by visual similarity with [MegaLoc](https://github.com/gmberton/MegaLoc), and verifies the most similar ones geometrically with [LightGlue](https://github.com/cvg/LightGlue) (ALIKED features) and RANSAC. The images never pass through the model; everything runs inside the tool.

- `image`: URL or local path of the image.
- `candidates`: list of `{"latitude": ..., "longitude": ...}` points, **sorted from most to least likely**. At most 50.
- `radius_m` (optional): search radius around each candidate, from 25 to 500 m. Default 150.

Candidates are processed in order, in batches of 10, and the search stops at the first `"match"`. For each candidate, up to 40 of the nearest Mapillary photos are kept: photos outside the circle, panoramas, and near-duplicates in the same sequence facing the same way are dropped. In each batch, the 20 photos most similar to the query are verified.

Returns:

- `matches`: only `"match"` and `"similar"` results, best first. Each has `candidate_index` (into your list), `reference_id` (the Mapillary image id, viewable at `https://www.mapillary.com/app/?pKey=<id>`), `reference_location` (latitude, longitude), `reference_heading_deg`, `reference_date` (ISO 8601, UTC), `inliers` and `verdict`.
- `candidates_checked`: how many candidates were processed.
- `candidates_without_coverage`: indices of candidates with no Mapillary photos nearby. They were **not ruled out**; they just could not be checked.
- `stopped_early`: `true` if the search stopped at a match.

Verdicts come from the RANSAC inliers of the LightGlue matches: `"match"` (60 or more) means the same place, geometrically verified; `"similar"` (20 to 59) is a weak lead; `"no_match"` is below 20. The thresholds are constants in `core/matching.py` and have not been tuned on a benchmark yet.

Requires a Mapillary token and the `verification` extra (see [Candidate verification setup](#candidate-verification-setup)). The first call downloads the MegaLoc code and weights (via `torch.hub` and Hugging Face) and the ALIKED and LightGlue weights. MegaLoc descriptors of Mapillary photos are cached in SQLite in the user cache directory (for example `~/Library/Caches/geotrace` on macOS or `~/.cache/geotrace` on Linux), so checking a place again is faster. On a CPU, a batch with no cached descriptors can take a minute or more.

### `compare_images`

Checks whether two images show the same place, by geometric verification only (LightGlue matches filtered with RANSAC). It uses the same function and thresholds as `verify_candidate_locations`.

- `image`, `reference`: URL or local path of each image.

Returns `inliers`, `matches` (LightGlue matches before RANSAC) and `verdict` (`"match"`, `"similar"` or `"no_match"`). Requires the `verification` extra, but no Mapillary token.

### Candidate verification setup

1. Install the extra in the GeoTrace directory:

   ```bash
   uv sync --extra verification
   ```

   (or `pip install -e ".[verification]"`). It installs PyTorch, torchvision, LightGlue (from its repository), OpenCV, safetensors and platformdirs. Without it, the rest of the server still works, and these two tools explain how to install it.

   All extras use the headless OpenCV build (`opencv-python-headless`), so the `calibration`, `ocr` and `verification` extras can be installed together. uv applies this automatically. pip cannot drop the `opencv-python` that LightGlue and GeoCalib require, and both packages ship the same `cv2` module. After a pip install of `verification` or `calibration`, run:

   ```bash
   pip uninstall -y opencv-python && pip install --force-reinstall --no-deps opencv-python-headless
   ```

2. Get a Mapillary token: sign in at [mapillary.com](https://www.mapillary.com), open the [developer dashboard](https://www.mapillary.com/dashboard/developers), register an application (read access is enough) and copy its **client access token** (it starts with `MLY|`).

3. Pass the token to the server in the `"env"` block of `claude_desktop_config.json`:

   ```json
   {
     "mcpServers": {
       "geotrace": {
         "command": "uv",
         "args": ["--directory", "/path/to/GeoTrace", "run", "geotrace"],
         "env": {
           "MAPILLARY_TOKEN": "MLY|..."
         }
       }
     }
   }
   ```

   Restart Claude Desktop afterwards. There is no default token; without one, `verify_candidate_locations` returns an error that explains these steps.

4. Optional: rank on a GPU in a Hugging Face Space instead of on this machine. MegaLoc is the slow step on a CPU, so the ranking can run in the [`davidlnk3/megaloc-descriptors`](https://huggingface.co/spaces/davidlnk3/megaloc-descriptors) Space (ZeroGPU), whose source is in `hf_space/megaloc/`. Add these to the same `"env"` block:

   ```json
   "GEOTRACE_RETRIEVAL_BACKEND": "space",
   "HF_TOKEN": "hf_..."
   ```

   `HF_TOKEN` (a read token from [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)) is optional. Without it, the calls use the small anonymous ZeroGPU quota. `GEOTRACE_MEGALOC_SPACE` selects another Space with the same API, for example your own duplicate. The Space receives the query image (re-encoded as a JPEG of at most 1024 px) and the 256 px Mapillary thumbnails whose descriptors are not cached yet. Do not use it for images that must not leave your machine. Both backends pin the same MegaLoc commit and share the local descriptor cache. LightGlue verification always runs locally, so the `verification` extra is still needed.

## License

GeoTrace is licensed under MIT License. See LICENSE.

## Third-party components

| Component | Used for | License |
|---|---|---|
| [SSIS / SSISv2](https://github.com/stevewongv/SSIS) | Object-shadow pair detection, run in a separate Hugging Face Space | **Non-commercial use only** |
| [GeoCalib](https://github.com/cvg/GeoCalib) | Camera calibration (gravity direction and focal length) to correct perspective in shadow measurements | Code: Apache 2.0 · Model weights: [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) |
| [All-Weather Deep Outdoor Lighting Estimation](https://openaccess.thecvf.com/content_CVPR_2019/html/Zhang_All-Weather_Deep_Outdoor_Lighting_Estimation_CVPR_2019_paper.html) | Sun-position inference as backup to object-shadow pair detection | Independent implementation, see below |
| [A Matter of Time](https://arxiv.org/abs/2510.19559) (Tekaya et al., 2025) and [EVA02-CLIP](https://github.com/baaivision/EVA) via [open_clip](https://github.com/mlfoundations/open_clip) | Time probing for `time_from_objects` | Method from the paper; weights downloaded at first use, not redistributed |
| [PLONK](https://arxiv.org/abs/2412.06781), [code](https://github.com/nicolas-dufour/plonk) | Regions and probabilities for `location_guesser_yfcc` (YFCC), `location_guesser_inat` (iNaturalist) and `location_guesser_osv` (OSV-5M) | Code: MIT · Weights: downloaded at first use from Hugging Face, not redistributed; check the model card licence |
| [reverse_geocoder](https://github.com/thampiman/reverse-geocoder) (GeoNames data) | Nearest place name, admin region and country code for `location_guesser_yfcc` | LGPL; data under GeoNames' CC BY 4.0 |
| [EasyOCR](https://github.com/JaidedAI/EasyOCR) (Jaided AI) | Text detection and recognition for `extract_text` (optional `ocr` extra) | Apache 2.0; models downloaded at first use, not redistributed |
| [OpenStreetMap](https://www.openstreetmap.org/copyright) data via the [Overpass API](https://wiki.openstreetmap.org/wiki/Overpass_API) | `overpass_query` and `find_named_features` | Data © OpenStreetMap contributors, [ODbL 1.0](https://opendatacommons.org/licenses/odbl/) |
| [Mapillary](https://www.mapillary.com) | Street-level reference photos for `verify_candidate_locations`, fetched at run time through the Mapillary API | Imagery: [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/); API use under the Mapillary terms |
| [MegaLoc](https://github.com/gmberton/MegaLoc) (Berton & Masone, CVPRW 2025) | Ranking the Mapillary photos by visual similarity | Code: MIT · Weights ([gberton/MegaLoc](https://huggingface.co/gberton/MegaLoc)): MIT; downloaded at first use, not redistributed. Optionally run in the `davidlnk3/megaloc-descriptors` Space |
| [LightGlue](https://github.com/cvg/LightGlue) with [ALIKED](https://github.com/Shiaoming/ALIKED) features | Feature matching for the geometric verification in `verify_candidate_locations` and `compare_images` | LightGlue code and weights: Apache 2.0 · ALIKED: BSD 3-Clause · Weights downloaded at first use, not redistributed. SuperPoint, whose licence is non-commercial, is not used |

### OpenStreetMap

Map data returned by `overpass_query` and `find_named_features` is © OpenStreetMap contributors and available under the Open Database License (ODbL). See https://www.openstreetmap.org/copyright.

### SSIS

Shadow detection uses SSIS / SSISv2 (Wang et al., CVPR 2021, TPAMI 2022),
https://github.com/stevewongv/SSIS, which is licensed for non-commercial use only.
It runs in a separate Hugging Face Space (davidlnk3/ssis-demo) that keeps the
original license and includes modifications described there.
For commercial use of SSIS, contact its authors.

### GeoCalib

GeoCalib is used unmodified as a dependency. Its weights are downloaded from the
official source on first use and are not redistributed with GeoTrace.

### All-Weather Deep Outdoor Lighting Estimation

Reimplementation of the sun-position part of *All-Weather Deep Outdoor Lighting Estimation* (Zhang et al., CVPR 2019). It is an independent implementation, not the authors' code, and it departs from the paper in several ways: a DINOv2-small backbone, sun labels from astral and the brightest pixel instead of PanoNet estimates, and crops that keep the sun outside the frame.

### Mapillary, MegaLoc and LightGlue

`verify_candidate_locations` downloads Mapillary thumbnails at run time and does not store or redistribute them. Only MegaLoc descriptors (numbers, not images) are cached locally, keyed by Mapillary image id. Mapillary imagery is licensed under CC BY-SA 4.0: if you publish a Mapillary photo found by the tool, credit Mapillary and the contributor, and link the image (`https://www.mapillary.com/app/?pKey=<reference_id>`).

MegaLoc is loaded with `torch.hub` from `gmberton/MegaLoc` and LightGlue is installed from `cvg/LightGlue`, both pinned to a commit and used unmodified. LightGlue can also run with DISK features (Apache 2.0, as stated in the LightGlue README), selected in `core/matching.py`.

Berton, G., & Masone, C. (2025). MegaLoc: One Retrieval to Place Them All. CVPR Workshops.
Lindenberger, P., Sarlin, P.-E., & Pollefeys, M. (2023). LightGlue: Local Feature Matching at Light Speed. ICCV.
