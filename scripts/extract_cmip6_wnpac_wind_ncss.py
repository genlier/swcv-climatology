"""Extract a wind-based WNPAC index from CMIP6 monthly U850.

Frozen sign convention (positive = anticyclonic shear):
U850(22.5--32.5N, 110--140E) - U850(5--15N, 90--130E).
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
MANIFEST = ROOT / "data" / "manifests" / "cmip6_monthly_ua_esgf_file_manifest.csv"
SPANS = ROOT / "data" / "processed" / "cmip6_picontrol_common_spans.csv"
PARTS = ROOT / "data" / "processed" / "cmip6_wnpac_wind_parts"
SUBSETS = ROOT / "data" / "raw" / "cmip6_wnpac_wind_subsets"
FAILURE_DIR = ROOT / "data" / "processed" / "cmip6_wnpac_wind_failures"
TRANSIENT = ["historical", "hist-GHG", "hist-aer", "hist-nat"]


def file_years(filename: str) -> tuple[int, int] | None:
    match = re.search(r"_(\d{4})\d{2}-(\d{4})\d{2}\.nc$", str(filename))
    return (int(match.group(1)), int(match.group(2))) if match else None


def download(url: str, query: dict, target: Path) -> None:
    urls = [url]
    if url.startswith("http://"):
        urls.insert(0, "https://" + url[len("http://"):])
    error = None
    for attempt in range(5):
        candidate = urls[attempt % len(urls)]
        try:
            with requests.get(candidate, params=query, timeout=(30, 300), stream=True) as response:
                response.raise_for_status()
                content_type = response.headers.get("content-type", "")
                if "html" in content_type.lower():
                    raise RuntimeError(f"unexpected content type {content_type}")
                with target.open("wb") as handle:
                    for chunk in response.iter_content(1024 * 1024):
                        if chunk:
                            handle.write(chunk)
            return
        except Exception as exc:
            error = f"{candidate}: {exc}"
            if attempt < 4:
                time.sleep(2 ** attempt)
    raise RuntimeError(f"NCSS failed after 5 attempts: {error}")


def area_mean(field: xr.DataArray, south: float, north: float, west: float, east: float) -> xr.DataArray:
    lat = "lat" if "lat" in field.coords else "latitude"
    lon = "lon" if "lon" in field.coords else "longitude"
    q = field.where(
        (field[lat] >= south) & (field[lat] <= north)
        & (field[lon] >= west) & (field[lon] <= east), drop=True
    )
    weights = np.cos(np.deg2rad(q[lat])).where(q.notnull()).fillna(0)
    dims = [dim for dim in q.dims if dim != "time"]
    return q.weighted(weights).mean(dims, skipna=True)


def extract(group: pd.DataFrame, source: str, experiment: str, member: str, start: int, end: int,
            chunk_years: int) -> pd.DataFrame:
    pieces = []
    ordered = group.dropna(subset=["ncss_url"]).copy()
    ordered["span"] = ordered.filename.map(file_years)
    ordered = ordered.sort_values("filename")
    for row in ordered.itertuples(index=False):
        if row.span is None or row.span[1] < start or row.span[0] > end:
            continue
        first, last = max(start, row.span[0]), min(end, row.span[1])
        target_dir = SUBSETS / source / experiment / member
        target_dir.mkdir(parents=True, exist_ok=True)
        for chunk_first in range(first, last + 1, chunk_years):
            chunk_last = min(chunk_first + chunk_years - 1, last)
            target = target_dir / f"{Path(row.filename).stem}__wnpac_boxes__{chunk_first}_{chunk_last}.nc"
            if not target.exists():
                partial = target.with_suffix(".nc.part")
                query = {
                    "var": "ua", "time_start": f"{chunk_first:04d}-01-01T00:00:00Z",
                    # Day 30 is valid for both 360-day and Gregorian calendars and still includes December means.
                    "time_end": f"{chunk_last:04d}-12-30T23:59:59Z", "accept": "netcdf4",
                    "addLatLon": "true", "north": 32.5, "south": 5, "west": 90, "east": 140,
                    "vertCoord": 85000,
                }
                download(row.ncss_url, query, partial)
                partial.replace(target)
            time_coder = xr.coders.CFDatetimeCoder(use_cftime=True)
            with xr.open_dataset(target, decode_times=time_coder) as ds:
                field = ds.ua
                for plev in ["plev", "lev"]:
                    if plev in field.dims:
                        field = field.sel({plev: 85000}, method="nearest")
                north_u = area_mean(field, 22.5, 32.5, 110, 140).load()
                south_u = area_mean(field, 5, 15, 90, 130).load()
                pieces.append(pd.DataFrame({
                    "year": north_u.time.dt.year.values.astype(int),
                    "month": north_u.time.dt.month.values.astype(int),
                    "north_u850_ms": north_u.values.astype(float),
                    "south_u850_ms": south_u.values.astype(float),
                    "wnpac_u850_ms": (north_u - south_u).values.astype(float),
                }))
            print(f"  {chunk_first}-{chunk_last}", flush=True)
    if not pieces:
        raise RuntimeError("no NCSS files overlap requested span")
    return pd.concat(pieces).drop_duplicates(["year", "month"]).sort_values(["year", "month"])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+")
    parser.add_argument("--experiments", nargs="+", choices=TRANSIENT + ["piControl"], default=TRANSIENT + ["piControl"])
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--chunk-years", type=int, default=10)
    args = parser.parse_args()
    files = pd.read_csv(MANIFEST)
    files = files.loc[files.experiment.isin(args.experiments)]
    spans = pd.read_csv(SPANS)
    if args.models:
        files = files.loc[files.source.isin(args.models)]
        spans = spans.loc[spans.source.isin(args.models)]
    PARTS.mkdir(parents=True, exist_ok=True)
    failures = []
    groups = files.groupby(["source", "experiment", "member"], sort=True)
    for index, ((source, experiment, member), group) in enumerate(groups, 1):
        if experiment == "piControl":
            span = spans.loc[(spans.source == source) & (spans.member == member)]
            if span.empty:
                raise RuntimeError(f"No frozen control span for {source} {member}")
            start, end = int(span.iloc[0].recommended_first_300_start), int(span.iloc[0].recommended_first_300_end)
        else:
            start, end = 1950, 2014
        output = PARTS / f"{source}_{experiment}_{member}_wnpac_u850_{start}_{end}.csv.gz"
        if output.exists() and not args.force:
            print(f"skip {index}/{len(groups)} {output.name}", flush=True)
            continue
        print(f"extract {index}/{len(groups)} {source} {experiment} {start}-{end}", flush=True)
        try:
            result = extract(group, source, experiment, member, start, end, args.chunk_years)
            result.to_csv(output, index=False)
        except Exception as exc:
            failures.append({"source": source, "experiment": experiment, "member": member, "error": str(exc)})
            print(f"FAILED: {exc}", flush=True)
    FAILURE_DIR.mkdir(parents=True, exist_ok=True)
    run_label = "_".join(args.models) if args.models else "all_models"
    failure_path = FAILURE_DIR / f"{run_label}.csv"
    pd.DataFrame(failures, columns=["source", "experiment", "member", "error"]).to_csv(failure_path, index=False)
    print(f"groups={len(groups)} failures={len(failures)}", flush=True)


if __name__ == "__main__":
    main()
