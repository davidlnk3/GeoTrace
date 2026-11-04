<p align="center">
  <img src="assets/geotrace-banner.png" alt="GeoTrace — Geolocate • Chronolocate" width="100%">
</p>

<div align="center">

### MCP for open source content verification from images

**Get the position and the time of any picture**
</div>

## Tools

### `extract_metadata`

Extracts EXIF metadata, camera data and GPS from an image.

- `image`: URL or local path of the image.

### `view_image`

Returns the image so the model can describe its visual clues.

- `image`: URL or local path of the image.

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

Returns `image_size`, `languages`, `blocks`, `total_blocks` and `truncated`. Each block has `text`, `confidence`, `bbox` (`x1, y1, x2, y2` in pixels) and `relative_size` (block area / image area). Blocks are sorted by `confidence × √relative_size`, so large sign text comes before small print, and at most 50 are returned. Words on the same line (similar height and baseline, small gap) are merged, so `VIA` + `ROMA` becomes `VIA ROMA`.

The image is OCR'd at full resolution, downscaled only when its longest side is above 3000 px. Each detected box is also tried rotated by 90°, 180° and 270°. OCR can misread characters (0/O, 1/l, accents), so check the text against the image.

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
