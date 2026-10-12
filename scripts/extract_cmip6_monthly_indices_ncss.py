"""Extract compact 1950--2014 CMIP6 monthly mechanism indices via NCSS.

Only server-side spatial/vertical subsets are downloaded.  Each model,
experiment, and variable is checkpointed to a small CSV so interrupted runs
resume without repeating completed transfers.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import re
import time

import numpy as np
import pandas as pd
import requests
import xarray as xr


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "data" / "manifests" / "cmip6_monthly_esgf_file_manifest.csv"
PARTS = ROOT / "data" / "processed" / "cmip6_monthly_indices_parts"
SUBSETS = ROOT / "data" / "raw" / "cmip6_monthly_subsets"
OUT = ROOT / "data" / "processed"
TRANSIENT = ["historical", "hist-GHG", "hist-aer", "hist-nat"]
START, END = 1950, 2014


def file_years(filename: str) -> tuple[int, int] | None:
    match = re.search(r"_(\d{4})\d{2}-(\d{4})\d{2}\.nc$", str(filename))
    return (int(match.group(1)), int(match.group(2))) if match else None


def spatial_mean(field: xr.DataArray) -> xr.DataArray:
    lat_name = "lat" if "lat" in field.coords else "latitude"
    lat = field[lat_name]
    weights = np.cos(np.deg2rad(lat)).where(field.notnull()).fillna(0)
    spatial_dims = [d for d in field.dims if d != "time"]
    return field.weighted(weights).mean(spatial_dims, skipna=True)


def ncss_params(variable: str, first_year: int, last_year: int) -> dict:
    params = {
        "var": variable, "time_start": f"{first_year}-01-01T00:00:00Z",
        # Day 30 is valid for Gregorian, no-leap and 360-day calendars.
        "time_end": f"{last_year}-12-30T23:59:59Z", "accept": "netcdf4",
        "addLatLon": "true",
    }
    if variable == "zg":
        params.update(north=25, south=15, west=115, east=150, vertCoord=85000)
    elif variable == "tos":
        params.update(north=20, south=-20, west=40, east=100)
    # tas intentionally has no bounding box: GMST requires the global field.
    return params


def download_subset(url: str, params: dict, target: Path) -> None:
    error = None
    urls = [url]
    if url.startswith("http://"):
        urls.insert(0, "https://" + url[len("http://"):])
    for attempt in range(3):
        candidate_url = urls[attempt % len(urls)]
        try:
            with requests.get(candidate_url, params=params, timeout=(30, 300), stream=True) as response:
                response.raise_for_status()
                ctype = response.headers.get("content-type", "")
                if "netcdf" not in ctype and "octet-stream" not in ctype:
                    raise RuntimeError(f"unexpected content type {ctype}: {response.text[:200]}")
                with target.open("wb") as handle:
                    for chunk in response.iter_content(1024 * 1024):
                        if chunk:
                            handle.write(chunk)
            return
        except Exception as exc:
            error = f"{candidate_url}: {exc}"
            if attempt < 2:
                time.sleep(2 ** attempt)
    raise RuntimeError(f"NCSS failed after 3 attempts: {error}")


def extract_group(group: pd.DataFrame, output: Path, chunk_years: int) -> None:
    variable = group.variable.iloc[0]
    pieces = []
    for row in group.sort_values("filename").itertuples():
        years = file_years(row.filename)
        if years is None or years[1] < START or years[0] > END:
            continue
        first, last = max(START, years[0]), min(END, years[1])
        if not isinstance(row.ncss_url, str) or not row.ncss_url:
            raise RuntimeError(f"No NCSS endpoint for {row.filename}")
        target_dir = SUBSETS / row.source / row.experiment / row.member / variable
        target_dir.mkdir(parents=True, exist_ok=True)
        for chunk_first in range(first, last + 1, chunk_years):
            chunk_last = min(chunk_first + chunk_years - 1, last)
            target = target_dir / f"{Path(row.filename).stem}__{chunk_first}_{chunk_last}.nc"
            if not target.exists():
                partial = target.with_suffix(target.suffix + ".part")
                download_subset(row.ncss_url, ncss_params(variable, chunk_first, chunk_last), partial)
                partial.replace(target)
            with xr.open_dataset(target) as ds:
                field = ds[variable]
                if variable == "zg" and "plev" in field.dims:
                    field = field.sel(plev=85000, method="nearest")
                mean = spatial_mean(field).load()
                dates = pd.DataFrame({
                    "year": mean.time.dt.year.values.astype(int),
                    "month": mean.time.dt.month.values.astype(int),
                    "value": mean.values.astype(float),
                })
                pieces.append(dates)
            print(f"  {variable}: {row.filename} ({chunk_first}-{chunk_last})", flush=True)
    if not pieces:
        raise RuntimeError(f"No 1950-2014 files for {group.iloc[0].source} {group.iloc[0].experiment} {variable}")
    data = pd.concat(pieces, ignore_index=True).drop_duplicates(["year", "month"]).sort_values(["year", "month"])
    data.to_csv(output, index=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="*", help="Optional source_id subset")
    parser.add_argument("--variables", nargs="+", choices=["tas", "tos", "zg"],
                        default=["tas", "tos", "zg"])
    parser.add_argument("--chunk-years", type=int, default=5)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    files = pd.read_csv(MANIFEST)
    files = files.loc[files.experiment.isin(TRANSIENT) & files.variable.isin(args.variables)].copy()
    if args.models:
        files = files.loc[files.source.isin(args.models)]
    PARTS.mkdir(parents=True, exist_ok=True); SUBSETS.mkdir(parents=True, exist_ok=True)
    failures = []
    groups = files.groupby(["source", "experiment", "member", "variable"], sort=True)
    for i, (key, group) in enumerate(groups, 1):
        source, experiment, member, variable = key
        output = PARTS / f"{source}_{experiment}_{member}_{variable}_1950_2014.csv.gz"
        if output.exists() and not args.force:
            print(f"skip {i}/{len(groups)} {output.name}", flush=True)
            continue
        print(f"extract {i}/{len(groups)} {source} {experiment} {member} {variable}", flush=True)
        try:
            extract_group(group, output, args.chunk_years)
        except Exception as exc:
            failures.append({"source": source, "experiment": experiment, "member": member,
                             "variable": variable, "error": str(exc)})
            print(f"FAILED: {exc}", flush=True)
    pd.DataFrame(failures).to_csv(OUT / "cmip6_monthly_ncss_failures.csv", index=False)
    print(f"complete groups={len(groups)-len(failures)} failures={len(failures)}", flush=True)


if __name__ == "__main__":
    main()
