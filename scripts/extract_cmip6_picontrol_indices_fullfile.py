"""Full-file fallback for piControl tas/tos when NCSS or Zarr spans do not align."""

from __future__ import annotations

import argparse
from pathlib import Path
import re

import numpy as np
import pandas as pd
import requests
import xarray as xr


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "data" / "manifests" / "cmip6_monthly_esgf_file_manifest.csv"
SPANS = ROOT / "data" / "processed" / "cmip6_picontrol_common_spans.csv"
RAW = ROOT / "data" / "raw" / "cmip6_picontrol_fullfiles"
PARTS = ROOT / "data" / "processed" / "cmip6_picontrol_indices_parts"


def file_years(filename: str) -> tuple[int, int] | None:
    match = re.search(r"_(\d{4})\d{2}-(\d{4})\d{2}\.nc$", filename)
    return (int(match.group(1)), int(match.group(2))) if match else None


def coordinate_names(field: xr.DataArray, variable: str) -> tuple[str, str]:
    if variable == "tos" and "latitude" in field.coords and field.latitude.ndim == 2:
        return "latitude", "longitude"
    if "nav_lat" in field.coords:
        return "nav_lat", "nav_lon"
    if "lat" in field.coords:
        return "lat", "lon"
    return "latitude", "longitude"


def area_mean(field: xr.DataArray, variable: str) -> xr.DataArray:
    lat_name, lon_name = coordinate_names(field, variable)
    lat, lon = field[lat_name], field[lon_name] % 360
    if variable == "tos":
        mask = (lat >= -20) & (lat <= 20) & (lon >= 40) & (lon <= 100)
        field = field.where(mask)
        weights = np.cos(np.deg2rad(lat)).where(mask, 0).fillna(0)
    else:
        weights = np.cos(np.deg2rad(lat)).fillna(0)
    dims = [dim for dim in field.dims if dim != "time"]
    return field.weighted(weights).mean(dims, skipna=True)


def download(url: str, target: Path, expected_size: int) -> None:
    if target.exists() and target.stat().st_size == expected_size:
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_suffix(".nc.part")
    candidate = "https://" + url[len("http://"):] if url.startswith("http://") else url
    for attempt in range(12):
        offset = partial.stat().st_size if partial.exists() else 0
        if offset >= expected_size:
            break
        headers = {"Range": f"bytes={offset}-"} if offset else {}
        try:
            with requests.get(candidate, headers=headers, timeout=(30, 60), stream=True) as response:
                response.raise_for_status()
                resume = offset > 0 and response.status_code == 206
                mode = "ab" if resume else "wb"
                if not resume:
                    offset = 0
                with partial.open(mode) as handle:
                    for chunk in response.iter_content(4 * 1024 * 1024):
                        if chunk:
                            handle.write(chunk)
            print(f"  received {partial.stat().st_size / 1024**2:.1f} MiB", flush=True)
        except requests.RequestException as exc:
            size = partial.stat().st_size / 1024**2 if partial.exists() else 0
            print(f"  retry {attempt + 1}/12 after {size:.1f} MiB: {exc}", flush=True)
    if not partial.exists() or partial.stat().st_size < expected_size:
        raise RuntimeError("download retry limit reached")
    if partial.stat().st_size != expected_size:
        raise RuntimeError(f"size mismatch: {partial.stat().st_size} != {expected_size}")
    partial.replace(target)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+")
    parser.add_argument("--variables", nargs="+", choices=["tas", "tos"], default=["tas", "tos"])
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    files = pd.read_csv(MANIFEST)
    files = files.loc[files.experiment.eq("piControl") & files.variable.isin(args.variables)]
    spans = pd.read_csv(SPANS)
    if args.models:
        files = files.loc[files.source.isin(args.models)]
        spans = spans.loc[spans.source.isin(args.models)]
    for (source, member, variable), group in files.groupby(["source", "member", "variable"]):
        span = spans.loc[(spans.source == source) & (spans.member == member)].iloc[0]
        start, end = int(span.recommended_first_300_start), int(span.recommended_first_300_end)
        output = PARTS / f"{source}_piControl_{member}_{variable}_{start}_{end}.csv.gz"
        if output.exists() and not args.force:
            print(f"skip {output.name}", flush=True)
            continue
        pieces = []
        for row in group.itertuples(index=False):
            years = file_years(row.filename)
            if years is None or years[1] < start or years[0] > end:
                continue
            target = RAW / source / member / variable / row.filename
            print(f"download {row.filename} ({row.file_size_bytes / 1024**3:.2f} GiB)", flush=True)
            download(row.http_url, target, int(row.file_size_bytes))
            with xr.open_dataset(target, chunks={"time": 120}, use_cftime=True) as ds:
                field = ds[variable]
                field = field.where((field.time.dt.year >= start) & (field.time.dt.year <= end), drop=True)
                mean = area_mean(field, variable).compute()
                pieces.append(pd.DataFrame({
                    "year": mean.time.dt.year.values.astype(int),
                    "month": mean.time.dt.month.values.astype(int),
                    "value": mean.values.astype(float),
                }))
        if not pieces:
            raise RuntimeError(f"No full files overlap {source} {variable} {start}-{end}")
        result = pd.concat(pieces).drop_duplicates(["year", "month"]).sort_values(["year", "month"])
        result.to_csv(output, index=False)
        print(f"wrote {output.name}: {len(result)} months", flush=True)


if __name__ == "__main__":
    main()
