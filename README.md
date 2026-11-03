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

### `detect_shadows`

Detects object-shadow pairs in an image and computes the sun elevation and the shadow direction of each pair. The camera is calibrated with GeoCalib, and the sun elevation is computed from that calibration. Requires the `calibration` extra: `pip install -e ".[calibration]"`.

- `image`: URL or local path of the image.

Returns:

- `camera`: the calibration (`roll_deg`, `pitch_deg`, `vertical_fov_deg`, `pitch_uncertainty_deg`, `vertical_fov_uncertainty_deg`).
- `num_pairs`, `pairs`: each pair has `pair_score` (detection confidence), `shadow_direction_deg` (on the ground, clockwise seen from above from the camera's forward direction: 0 = away from the camera, 90 = right) and `sun_elevation_deg`.
- `skipped`: pairs that could not be computed, with the reason (for example, a base or tip above the horizon).

### `shadow_to_time`

Given a location and the sun's position, returns the dates and times (UTC) when the sun was in that position.

- `latitude`, `longitude`: decimal degrees.
- `sun_elevation`: degrees above the horizon, greater than 0.
- `sun_azimuth`: degrees clockwise from north. A shadow points opposite the sun, so `sun_azimuth = shadow_direction + 180` (mod 360).
- `year` (optional): year to search. Defaults to the current year.
- `tolerance` (optional): maximum angular error in degrees. Defaults to 1.
- `max_results` (optional): maximum number of candidates. Defaults to 10.

Returns the candidates sorted by error, with the date and time in UTC.

### `shadow_to_location`

Given the date and time of a photo and the sun's position, returns the locations where the sun was in that position.

- `date`: UTC date, `YYYY-MM-DD`.
- `time`: UTC time, `HH:MM` or `HH:MM:SS`.
- `sun_elevation`: degrees above the horizon, greater than 0.
- `sun_azimuth`: degrees clockwise from north.

Usually two mirror-image solutions exist; both are returned, each with latitude and longitude.

### `estimate_sun`

Estimates the sun's altitude and its azimuth relative to the camera from a single outdoor image, with a DINOv2-small model trained on crops where the sun is outside the frame. It uses the centre square of the image.

- `image`: URL or local path of the image.

Returns:

- `sun_altitude_deg`: sun elevation above the horizon, 0 to 90.
- `sun_azimuth_relative_camera_deg`: sun direction clockwise from the camera's forward direction, in [0, 360). 0 = straight ahead, 90 = right, 180 = behind.

The model reads the sun from scene cues (shadows, lighting, sky brightness), so it is an estimate. On the validation set of the training project, the median errors were 1.6° for altitude and 5.6° for azimuth, with a long tail of large azimuth errors.

Requires `torch` and `transformers`, which are not in the default dependencies. Install them in the GeoTrace environment with `pip install torch transformers`. The model weights are in the repository at `models/sun/best.pt`, so the tool needs no external files.

### `time_from_objects`

Lists the human-made objects in an image and the interval in which each was built, from its likely build year until today, with EVA02-CLIP time probing as in [arXiv:2510.19559](https://arxiv.org/abs/2510.19559). It helps narrow when the photo can have been taken: it cannot be older than the most recent object in it.

- `image`: URL or local path of the image.
- `objects` (optional): object names to date, for example `["bicycle", "street lamp"]`. If omitted, the TIME10k classes (car, mobile phone, ship, musical instrument, aircraft, weapon or ammunition) are checked for presence.

Returns:

- `current_year`, `image_taken_not_before_year`: the latest `built_from_year` among the objects.
- `objects`: for each object, `most_likely_build_year`, `built_from_year`, `built_until_year` (today), `tolerance_years`, and `presence` (zero-shot probability, only when objects are detected).

`built_from_year` is the most likely build year minus the paper's accuracy tolerance for that year (20 years at 1700, 5 years at 2024). The paper scores whole images with prompts that name no object; adding the object name to the prompt, and the interval rule, are this project's extensions and are not evaluated in the paper.

Requires `torch`, `transformers` and `open_clip_torch`: `pip install torch open_clip_torch`. The EVA02-CLIP weights (`EVA02-L-14-336`, `merged2b_s6b_b61k`) are downloaded on first use.

## License

GeoTrace is licensed under MIT License. See LICENSE.

## Third-party components

| Component | Used for | License |
|---|---|---|
| [SSIS / SSISv2](https://github.com/stevewongv/SSIS) | Object-shadow pair detection, run in a separate Hugging Face Space | **Non-commercial use only** |
| [GeoCalib](https://github.com/cvg/GeoCalib) | Camera calibration (gravity direction and focal length) to correct perspective in shadow measurements | Code: Apache 2.0 · Model weights: [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) |
| [All-Weather Deep Outdoor Lighting Estimation](https://openaccess.thecvf.com/content_CVPR_2019/html/Zhang_All-Weather_Deep_Outdoor_Lighting_Estimation_CVPR_2019_paper.html) (Zhang et al., CVPR 2019) | Reimplementation of the sun-position part, used by `estimate_sun` | Independent implementation, see below |
| [A Matter of Time](https://arxiv.org/abs/2510.19559) (Tekaya et al., 2025) and [EVA02-CLIP](https://github.com/baaivision/EVA) via [open_clip](https://github.com/mlfoundations/open_clip) | Time probing for `time_from_objects` | Method from the paper; weights downloaded at first use, not redistributed |

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

`estimate_sun` is a reimplementation of the sun-position part of *All-Weather Deep Outdoor Lighting Estimation* (Zhang et al., CVPR 2019). It is an independent implementation, not the authors' code, and it departs from the paper in several ways: a DINOv2-small backbone, sun labels from astral and the brightest pixel instead of PanoNet estimates, and crops that keep the sun outside the frame.
