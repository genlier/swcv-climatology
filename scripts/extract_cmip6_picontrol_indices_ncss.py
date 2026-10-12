"""Extract the first 300 common piControl years through ESGF NCSS."""

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
SPANS = ROOT / "data" / "processed" / "cmip6_picontrol_common_spans.csv"
PARTS = ROOT / "data" / "processed" / "cmip6_picontrol_indices_parts"
SUBSETS = ROOT / "data" / "raw" / "cmip6_picontrol_subsets"
OUT = ROOT / "data" / "processed"


def file_years(filename: str) -> tuple[int, int] | None:
    match = re.search(r"_(\d{4})\d{2}-(\d{4})\d{2}\.nc$", str(filename))
    return (int(match.group(1)), int(match.group(2))) if match else None


def spatial_mean(field: xr.DataArray) -> xr.DataArray:
    lat_name = next(name for name in ("lat", "latitude", "nav_lat") if name in field.coords)
    weights = np.cos(np.deg2rad(field[lat_name])).where(field.notnull()).fillna(0)
    dims = [dim for dim in field.dims if dim != "time"]
    return field.weighted(weights).mean(dims, skipna=True)


def params(variable: str, first: int, last: int) -> dict:
    result = {
        "var": variable,
        "time_start": f"{first:04d}-01-01T00:00:00Z",
        # Day 30 is valid for Gregorian, no-leap and 360-day calendars.
        "time_end": f"{last:04d}-12-30T23:59:59Z",
        "accept": "netcdf4",
        "addLatLon": "true",
    }
    if variable == "zg":
        result.update(north=25, south=15, west=115, east=150, vertCoord=85000)
    elif variable == "tos":
        result.update(north=20, south=-20, west=40, east=100)
    return result


def download(url: str, query: dict, target: Path) -> None:
    urls = [url]
    if url.startswith("http://"):
        urls.insert(0, "https://" + url[len("http://"):])
    error = None
    for attempt in range(3):
        candidate = urls[attempt % len(urls)]
        try:
            with requests.get(candidate, params=query, timeout=(30, 300), stream=True) as response:
                response.raise_for_status()
                with target.open("wb") as handle:
                    for chunk in response.iter_content(1024 * 1024):
                        if chunk:
                            handle.write(chunk)
            return
        except Exception as exc:
            error = f"{candidate}: {exc}"
            if attempt < 2:
                time.sleep(2 ** attempt)
    raise RuntimeError(f"NCSS failed after 3 attempts: {error}")


def extract(group: pd.DataFrame, source: str, member: str, variable: str, start: int,
            end: int, chunk_years: int) -> pd.DataFrame:
    pieces = []
    ordered = group.assign(span=group.filename.map(file_years)).sort_values("filename")
    for row in ordered.itertuples(index=False):
        if row.span is None or row.span[1] < start or row.span[0] > end:
            continue
        first, last = max(start, row.span[0]), min(end, row.span[1])
        target_dir = SUBSETS / source / member / variable
        target_dir.mkdir(parents=True, exist_ok=True)
        for chunk_first in range(first, last + 1, chunk_years):
            chunk_last = min(chunk_first + chunk_years - 1, last)
            target = target_dir / f"{Path(row.filename).stem}__{chunk_first}_{chunk_last}.nc"
            if not target.exists():
                partial = target.with_suffix(".nc.part")
                download(row.ncss_url, params(variable, chunk_first, chunk_last), partial)
                partial.replace(target)
            with xr.open_dataset(target) as ds:
                field = ds[variable]
                if variable == "zg" and "plev" in field.dims:
                    field = field.sel(plev=85000, method="nearest")
                mean = spatial_mean(field).load()
                pieces.append(pd.DataFrame({
                    "year": mean.time.dt.year.values.astype(int),
                    "month": mean.time.dt.month.values.astype(int),
                    "value": mean.values.astype(float),
                }))
            print(f"  {variable} {chunk_first}-{chunk_last}", flush=True)
    if not pieces:
        raise RuntimeError("no files overlap selected control span")
    return pd.concat(pieces).drop_duplicates(["year", "month"]).sort_values(["year", "month"])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+")
    parser.add_argument("--variables", nargs="+", choices=["tas", "tos", "zg"], default=["tas", "tos", "zg"])
    parser.add_argument("--chunk-years", type=int, default=5)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    files = pd.read_csv(MANIFEST)
    files = files.loc[files.experiment.eq("piControl") & files.variable.isin(args.variables)]
    spans = pd.read_csv(SPANS)
    if args.models:
        files = files.loc[files.source.isin(args.models)]
        spans = spans.loc[spans.source.isin(args.models)]
    PARTS.mkdir(parents=True, exist_ok=True)
    failures = []
    groups = files.groupby(["source", "member", "variable"], sort=True)
    for index, ((source, member, variable), group) in enumerate(groups, 1):
        span = spans.loc[(spans.source == source) & (spans.member == member)].iloc[0]
        start, end = int(span.recommended_first_300_start), int(span.recommended_first_300_end)
        output = PARTS / f"{source}_piControl_{member}_{variable}_{start}_{end}.csv.gz"
        if output.exists() and not args.force:
            print(f"skip {index}/{len(groups)} {output.name}", flush=True)
            continue
        print(f"extract {index}/{len(groups)} {source} {variable} {start}-{end}", flush=True)
        try:
            result = extract(group, source, member, variable, start, end, args.chunk_years)
            result.to_csv(output, index=False)
        except Exception as exc:
            failures.append({"source": source, "member": member, "variable": variable, "error": str(exc)})
            print(f"FAILED: {exc}", flush=True)
    pd.DataFrame(failures).to_csv(OUT / "cmip6_picontrol_ncss_failures.csv", index=False)
    print(f"groups={len(groups)} failures={len(failures)}", flush=True)


if __name__ == "__main__":
    main()
