"""Sample N photos from the DEW meta.csv and download them from Flickr.

DEW ("Date Estimation in the Wild", Mueller et al. 2017, doi:10.22000/43) ships only a CSV of
Flickr URLs with their capture year; the images are fetched from Flickr. Licences vary per photo
and plenty are "All Rights Reserved", so each row's licence is recorded in the manifest and the
images are used to fit coefficients, never redistributed.

meta.csv columns: img_id, GT (year), date_taken, date_granularity, url, username, title, license,
license_url

The sample is spread evenly over the year classes rather than drawn uniformly, so the extra images
cover the whole 1950-1999 range instead of following DEW's own skew. meta.csv is sorted by year,
so an even spread also makes the sample robust to a partial download.

Pass --exclude-csv with DEW's own test/validation id lists to keep its benchmark images out of
training, so this model can still be compared against DEW's published numbers.

Writes manifest.json as [{"file": ..., "year": ..., "flickr_id": ..., "license": ...}, ...].
"""
import argparse
import csv
import json
import random
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

YEAR_MIN, YEAR_MAX = 1950, 1999
TIMEOUT = 15

# DEW lists 1,029,710 Flickr images but states only 155,971 carry open-access licences; the rest
# are "All Rights Reserved" and listed by URL alone. Sample only the open ones by default -- there
# are ~180k of them, far more than needed.
OPEN_LICENCES = frozenset({
    "Attribution License",
    "Attribution-ShareAlike License",
    "Attribution-NonCommercial License",
    "Attribution-NonCommercial-ShareAlike License",
    "Attribution-NoDerivs License",
    "Attribution-NonCommercial-NoDerivs License",
    "No known copyright restrictions",
    "Public Domain Mark",
    "Public Domain Dedication (CC0)",
    "United States Government Work",
})
# A generic client name: the dataset's own downloader sends no contact details either.
HEADERS = {"User-Agent": "dew-sample/1.0"}


def read_ids(paths: list[str]) -> set[str]:
    """Collect img_ids from DEW's split CSVs, which are headed `GT,img_id`.

    The id is NOT the first column, so the column is located by name; reading column 0 silently
    collects the years instead and yields a filter that matches nothing.
    """
    ids: set[str] = set()
    for path in paths:
        with open(path, newline="", encoding="utf-8", errors="replace") as fh:
            rows = csv.reader(fh)
            header = next(rows, None)
            if header is None:
                continue
            names = [h.strip().lower() for h in header]
            col = names.index("img_id") if "img_id" in names else len(names) - 1
            if header[col].strip().isdigit():       # no header after all: that row is data
                ids.add(header[col].strip())
            for row in rows:
                if len(row) > col and row[col].strip().isdigit():
                    ids.add(row[col].strip())
    return ids


def fetch(url: str) -> bytes | None:
    try:
        with urlopen(Request(url, headers=HEADERS), timeout=TIMEOUT) as fh:
            return fh.read()
    except (HTTPError, URLError, OSError) as exc:
        print(f"  miss {url}: {type(exc).__name__}: {exc}", flush=True)
        return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--meta", required=True, help="path to DEW meta.csv")
    ap.add_argument("--out", required=True, help="output folder for images + manifest.json")
    ap.add_argument("--n", type=int, default=500)
    ap.add_argument("--seed", type=int, default=20)   # the paper's shuffle seed
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--exclude-csv", nargs="*", default=[],
                    help="CSVs whose first column lists img_ids to keep out of the sample")
    ap.add_argument("--any-licence", action="store_true",
                    help="also sample All Rights Reserved photos (default: open licences only)")
    ap.add_argument("--only-csv", nargs="*", default=[],
                    help="restrict the sample to img_ids listed in these CSVs (held-out eval set)")
    args = ap.parse_args()

    only = read_ids(args.only_csv)
    if only:
        print(f"restricted to {len(only)} ids", flush=True)

    excluded = read_ids(args.exclude_csv)
    if excluded:
        print(f"excluding {len(excluded)} ids from DEW's own splits", flush=True)

    out = Path(args.out)
    (out / "images").mkdir(parents=True, exist_ok=True)

    by_year: dict[int, list[tuple[str, str, str]]] = {}
    skipped_licence = 0
    with open(args.meta, newline="", encoding="utf-8", errors="replace") as fh:
        reader = csv.reader(fh)
        for row in reader:
            if len(row) < 5 or not row[1].strip().isdigit():
                continue                                      # header or malformed line
            year = int(row[1])
            licence = row[7].strip() if len(row) > 7 else ""
            if not (YEAR_MIN <= year <= YEAR_MAX) or row[0].strip() in excluded:
                continue
            if only and row[0].strip() not in only:
                continue
            if not args.any_licence and licence not in OPEN_LICENCES:
                skipped_licence += 1
                continue
            by_year.setdefault(year, []).append((row[0], row[4], licence))

    years = sorted(by_year)
    print(f"meta.csv: {sum(len(v) for v in by_year.values())} usable rows in {YEAR_MIN}-{YEAR_MAX} "
          f"across {len(years)} years "
          f"({skipped_licence} skipped as not openly licensed)", flush=True)
    if not years:
        print("no usable rows; is this the DEW meta.csv?")
        return 1

    rng = random.Random(args.seed)
    for v in by_year.values():
        rng.shuffle(v)

    # Round-robin over years, oversampling so download failures still leave us at n.
    queue: list[tuple[int, str, str, str]] = []
    depth = 0
    while len(queue) < args.n * 3:
        added = False
        for y in years:
            if depth < len(by_year[y]):
                fid, url, lic = by_year[y][depth]
                queue.append((y, fid, url, lic))
                added = True
        if not added:
            break
        depth += 1

    manifest: list[dict] = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for i in range(0, len(queue), args.workers * 4):
            if len(manifest) >= args.n:
                break
            chunk = queue[i:i + args.workers * 4]
            for (year, fid, url, lic), raw in zip(chunk, pool.map(lambda c: fetch(c[2]), chunk)):
                if raw is None or len(raw) < 1024:
                    continue
                name = f"{fid}.jpg"
                (out / "images" / name).write_bytes(raw)
                manifest.append({"file": name, "year": year, "flickr_id": fid, "license": lic})
                if len(manifest) >= args.n:
                    break
            print(f"  {len(manifest)}/{args.n} downloaded", flush=True)

    with open(out / "manifest.json", "w") as fh:
        json.dump(manifest, fh, indent=1)
    got = {}
    for m in manifest:
        got[m["year"]] = got.get(m["year"], 0) + 1
    print(f"wrote {len(manifest)} images and manifest.json; {len(got)} distinct years covered")
    return 0 if manifest else 1


if __name__ == "__main__":
    sys.exit(main())
